from typing import Callable, Dict, Iterable, List, Optional, Union

import asyncio
import inspect
import logging
import os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from functools import partial
from urllib.parse import quote

from attrs import define, field
from tqdm import tqdm

from parea import Parea
from parea.constants import PAREA_OS_ENV_EXPERIMENT_UUID
from parea.experiment.dvc import save_results_to_dvc_if_init
from parea.helpers import duplicate_dicts, gen_random_name, is_logging_disabled
from parea.schemas import EvaluationResult
from parea.schemas.models import CreateExperimentRequest, ExperimentSchema, ExperimentStatsSchema, ExperimentStatus, FinishExperimentRequestSchema
from parea.utils.trace_utils import thread_ids_running_evals, trace_data
from parea.utils.universal_encoder import json_dumps

STAT_ATTRS = ["latency", "input_tokens", "output_tokens", "total_tokens", "cost"]
logger = logging.getLogger()


def calculate_avg_std_for_experiment(experiment_stats: ExperimentStatsSchema) -> Dict[str, str]:
    accumulators = defaultdict(float)
    counts = defaultdict(int)
    score_accumulators = defaultdict(float)
    score_counts = defaultdict(int)

    for trace_stat in experiment_stats.parent_trace_stats:
        for attr in STAT_ATTRS:
            value = getattr(trace_stat, attr, None)
            if value is not None:
                accumulators[attr] += value
                counts[attr] += 1

        for score in trace_stat.scores:
            score_accumulators[score.name] += score.score
            score_counts[score.name] += 1

    averages = {attr: "N/A" if counts[attr] == 0 else f"{accumulators[attr] / counts[attr]:.{5 if attr == 'cost' else 2}f}" for attr in accumulators}

    score_averages = {f"{name}": "N/A" if score_counts[name] == 0 else f"{score_accumulators[name] / score_counts[name]:.2f}" for name in score_accumulators}

    averages.update(score_averages)

    return averages


def async_wrapper(fn, **kwargs):
    return asyncio.run(fn(**kwargs))


def apply_dataset_eval(dataset_level_evals: List[Callable]) -> List[EvaluationResult]:
    root_traces = []
    for trace in trace_data.get().values():
        if trace.root_trace_id == trace.trace_id:
            root_traces.append(trace)

    results = []
    for dataset_level_eval in dataset_level_evals:
        try:
            result = dataset_level_eval(root_traces)
        except Exception as e:
            logger.error(f"Error occurred calling dataset level eval function '{dataset_level_eval.__name__}': {e}", exc_info=e)
            continue
        if result is None:
            continue

        if isinstance(result, EvaluationResult):
            results.append(result)
        elif isinstance(result, list):
            results.extend(result)
        else:
            results.append(EvaluationResult(name=dataset_level_eval.__name__, score=result))

    return results


async def experiment(
    experiment_name: str,
    run_name: str,
    data: Union[str, int, Iterable[dict]],
    func: Callable,
    p: Parea,
    experiment_uuid: str,
    n_trials: int = 1,
    dataset_level_evals: Optional[List[Callable]] = None,
    n_workers: int = 10,
    stop_on_error: bool = True,
) -> ExperimentStatsSchema:
    """Creates an experiment and runs the function on the data iterator.
    param experiment_name: The name of the experiment. Used to organize experiments within a project.
    param run_name: The run name of the experiment. This name must be unique across experiment runs.
    param data: The data to run the experiment on. This can be a list of dictionaries,
        a string representing the name of a dataset on Parea or an int representing the id of a dataset on Parea.
        If it is a list of dictionaries, the key "target" is reserved for the target/expected output of that sample.
    param func: The function to run. This function should accept inputs that match the keys of the data field.
    param p: The Parea instance to use for running the experiment.
    param experiment_uuid: The UUID of the experiment. This is used to associate traces with the experiment.
    param n_trials: The number of times to run the experiment on the same data.
    param dataset_level_evals: A list of functions to run on the dataset level. Each function should accept a list of EvaluatedLogs and return a float or a
        EvaluationResult. If a float is returned, the name of the function will be used as the name of the evaluation.
    param n_workers: The number of workers to use for running the experiment.
    param stop_on_error: If True, the experiment will stop running if an exception is raised.
    """
    if isinstance(data, (str, int)):
        print(f"Fetching test collection: {data}")
        test_collection = await p.aget_collection(data)
        len_test_cases = test_collection.num_test_cases()
        print(f"Fetched {test_collection.num_test_cases()} test cases from collection: {data} \n")
        data: Iterable[dict] = test_collection.get_all_test_inputs_and_targets_dict()
    else:
        len_test_cases = len(data) if isinstance(data, list) else 0

    if n_trials > 1:
        data = duplicate_dicts(data, n_trials)
        len_test_cases = len(data) if isinstance(data, list) else 0
        print(f"Running {n_trials} trials of the experiment \n")

    os.environ[PAREA_OS_ENV_EXPERIMENT_UUID] = experiment_uuid

    sem = asyncio.Semaphore(n_workers)

    async def limit_concurrency(sample):
        async with sem:
            sample_copy = deepcopy(sample)
            target = sample_copy.pop("target", None)
            return await func(_parea_target_field=target, **sample_copy)

    def limit_concurrency_sync(sample):
        sample_copy = deepcopy(sample)
        target = sample_copy.pop("target", None)
        return func(_parea_target_field=target, **sample_copy)

    if inspect.iscoroutinefunction(func):
        tasks = [asyncio.ensure_future(limit_concurrency(sample)) for sample in data]
    else:
        executor = ThreadPoolExecutor(max_workers=n_workers)
        loop = asyncio.get_event_loop()
        tasks = [asyncio.ensure_future(loop.run_in_executor(executor, partial(limit_concurrency_sync, sample))) for sample in data]

    status = ExperimentStatus.COMPLETED
    with tqdm(total=len(tasks), desc="Running samples", unit="sample") as pbar:
        try:
            for coro in asyncio.as_completed(tasks):
                try:
                    await coro
                    pbar.update(1)
                except Exception as e:
                    status = ExperimentStatus.FAILED
                    if stop_on_error:
                        print(f"\nExperiment stopped due to an error (note you can deactivate this behavior by setting stop_on_error=False): {str(e)}\n")
                        for task in tasks:
                            task.cancel()
                    else:
                        pbar.update(1)
        except asyncio.CancelledError:
            pass

    await asyncio.sleep(0.2)
    total_evals = len(thread_ids_running_evals.get())
    with tqdm(total=total_evals, dynamic_ncols=True) as pbar:
        while thread_ids_running_evals.get():
            pbar.set_description("Waiting for evaluations to finish")
            pbar.update(total_evals - len(thread_ids_running_evals.get()))
            total_evals = len(thread_ids_running_evals.get())
            await asyncio.sleep(0.5)
        await asyncio.sleep(4)
        pbar.update(total_evals)

    if dataset_level_evals:
        dataset_level_eval_results = apply_dataset_eval(dataset_level_evals)
    else:
        dataset_level_eval_results = []

    experiment_stats: ExperimentStatsSchema = p.finish_experiment(experiment_uuid, FinishExperimentRequestSchema(status=status, dataset_level_stats=dataset_level_eval_results))
    stat_name_to_avg_std = calculate_avg_std_for_experiment(experiment_stats)
    if dataset_level_eval_results:
        stat_name_to_avg_std.update(**{eval_result.name: eval_result.score for eval_result in dataset_level_eval_results})
    print(f"Experiment {experiment_name} Run {run_name} stats:\n{json_dumps(stat_name_to_avg_std, indent=2)}\n\n")
    print(f"View experiment & traces at: https://app.parea.ai/experiments/{quote(experiment_name, safe='')}/{experiment_uuid}\n")
    save_results_to_dvc_if_init(run_name, stat_name_to_avg_std)

    if os.environ.get(PAREA_OS_ENV_EXPERIMENT_UUID, None):
        del os.environ[PAREA_OS_ENV_EXPERIMENT_UUID]

    return experiment_stats


_experiments = []


def data_converter(data: Union[str, int, Iterable[dict]]) -> Union[str, int, Iterable[dict]]:
    if isinstance(data, (str, int)):
        return data
    else:
        for sample in data:
            if "target" in sample and isinstance(sample["target"], dict):
                sample["target"] = json_dumps(sample["target"])
        return data


@define
class Experiment:
    # If your dataset is defined locally it should be an iterable of k/v
    # pairs matching the expected inputs of your function. To reference a dataset you
    # have saved on Parea, use the dataset name as a string or the id as an int.
    data: Union[str, int, Iterable[dict]] = field(converter=data_converter)
    # The function to run. This function should accept inputs that match the keys of the data field.
    func: Callable = field()
    experiment_stats: ExperimentStatsSchema = field(init=False, default=None)
    metadata: Optional[Dict[str, str]] = field(default=None)
    dataset_level_evals: Optional[List[Callable]] = field(default=None)
    p: Parea = field(default=None)
    experiment_name: str = field
    run_name: str = field(init=False)
    experiment_uuid: str = field(init=False, default=None)
    n_workers: int = field(default=10)
    # The number of times to run the experiment on the same data.
    n_trials: int = field(default=1)
    stop_on_error: bool = field(default=True)

    def __attrs_post_init__(self):
        global _experiments
        _experiments.append(self)
        if isinstance(self.data, str):
            if self.metadata is None:
                self.metadata = {"Dataset": self.data}
            else:
                if "Dataset" in self.metadata:
                    raise ValueError("Metadata should not contain a key 'Dataset' when using uploaded dataset (data is a string).")
                self.metadata["Dataset"] = self.data

    def _gen_run_name_if_none(self, name: Optional[str]):
        if not name:
            self.run_name = gen_random_name()
            print(f"Run name set to: {self.run_name}, since a name was not provided.")
        else:
            self.run_name = name

    def run(self, run_name: Optional[str] = None) -> None:
        """Run the experiment and save the results to DVC.
        param run_name: The run name of the experiment. This name must be unique across experiment runs.
        If no run name is provided a memorable name will be generated automatically.
        """
        if is_logging_disabled():
            print("Parea logging is turned off. Experiment can't be run without logging. Set env var TURN_OFF_PAREA_LOGGING to False to enable.")
            return

        try:
            self._gen_run_name_if_none(run_name)
            experiment_schema: ExperimentSchema = self.p.create_experiment(CreateExperimentRequest(name=self.experiment_name, run_name=self.run_name, metadata=self.metadata))
            self.experiment_uuid = experiment_schema.uuid
            self.experiment_stats = asyncio.run(
                experiment(
                    self.experiment_name,
                    self.run_name,
                    self.data,
                    self.func,
                    self.p,
                    self.experiment_uuid,
                    self.n_trials,
                    self.dataset_level_evals,
                    self.n_workers,
                    self.stop_on_error,
                )
            )
        except Exception as e:
            import traceback

            traceback.print_exc()
            print(f"Error running experiment: {e}")

    async def arun(self, run_name: Optional[str] = None) -> None:
        """Run the experiment and save the results to DVC.
        param run_name: The run name of the experiment. This name must be unique across experiment runs.
        If no run name is provided a memorable name will be generated automatically.
        """
        if is_logging_disabled():
            print("Parea logging is turned off. Experiment can't be run without logging. Set env var TURN_OFF_PAREA_LOGGING to False to enable.")
            return

        try:
            self._gen_run_name_if_none(run_name)
            experiment_schema: ExperimentSchema = await self.p.acreate_experiment(
                CreateExperimentRequest(name=self.experiment_name, run_name=self.run_name, metadata=self.metadata)
            )
            self.experiment_uuid = experiment_schema.uuid
            self.experiment_stats = await experiment(
                self.experiment_name, self.run_name, self.data, self.func, self.p, self.experiment_uuid, self.n_trials, self.dataset_level_evals, self.n_workers, self.stop_on_error
            )
        except Exception as e:
            import traceback

            traceback.print_exc()
            print(f"Error running experiment: {e}")

    @property
    def avg_scores(self) -> Dict[str, float]:
        """Returns the average score across all evals."""
        return self.experiment_stats.avg_scores if self.experiment_stats else {}
