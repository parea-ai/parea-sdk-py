from typing import Callable, Optional, Union

import asyncio
import inspect
import os
import time
from collections import defaultdict
from collections.abc import Iterable

from attrs import define, field
from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio

from parea import Parea
from parea.constants import PAREA_OS_ENV_EXPERIMENT_UUID
from parea.experiment.dvc import save_results_to_dvc_if_init
from parea.helpers import gen_random_name
from parea.schemas.models import CreateExperimentRequest, ExperimentSchema, ExperimentStatsSchema
from parea.utils.trace_utils import thread_ids_running_evals
from parea.utils.universal_encoder import json_dumps

STAT_ATTRS = ["latency", "input_tokens", "output_tokens", "total_tokens", "cost"]


def calculate_avg_std_for_experiment(experiment_stats: ExperimentStatsSchema) -> dict[str, str]:
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

    averages = {attr: "N/A" if counts[attr] == 0 else f"{accumulators[attr] / counts[attr]:.2f}" for attr in accumulators}

    score_averages = {f"{name}": "N/A" if score_counts[name] == 0 else f"{score_accumulators[name] / score_counts[name]:.2f}" for name in score_accumulators}

    averages.update(score_averages)

    return averages


def async_wrapper(fn, **kwargs):
    return asyncio.run(fn(**kwargs))


async def experiment(name: str, data: Union[str, Iterable[dict]], func: Callable, p: Parea) -> ExperimentStatsSchema:
    """Creates an experiment and runs the function on the data iterator."""
    if isinstance(data, str):
        print(f"Fetching test collection: {data}")
        test_collection = await p.aget_collection(data)
        print(f"Fetched {test_collection.num_test_cases()} test cases from collection: {data} \n")
        data: Iterable[dict] = test_collection.get_all_test_case_inputs()

    experiment_schema: ExperimentSchema = p.create_experiment(CreateExperimentRequest(name=name))
    experiment_uuid = experiment_schema.uuid
    os.environ[PAREA_OS_ENV_EXPERIMENT_UUID] = experiment_uuid

    max_parallel_calls = 10
    sem = asyncio.Semaphore(max_parallel_calls)

    async def limit_concurrency(data_input):
        async with sem:
            return await func(**data_input)

    if inspect.iscoroutinefunction(func):
        tasks = [limit_concurrency(data_input) for data_input in data]
        for result in tqdm_asyncio(tasks):
            await result
    else:
        for data_input in tqdm(data):
            func(**data_input)

    total_evals = len(thread_ids_running_evals.get())
    with tqdm(total=total_evals, dynamic_ncols=True) as pbar:
        while thread_ids_running_evals.get():
            pbar.set_description(f"Waiting for evaluations to finish")
            pbar.update(total_evals - len(thread_ids_running_evals.get()))
            total_evals = len(thread_ids_running_evals.get())
            await asyncio.sleep(0.5)

    time.sleep(2)
    experiment_stats: ExperimentStatsSchema = p.finish_experiment(experiment_uuid)
    stat_name_to_avg_std = calculate_avg_std_for_experiment(experiment_stats)
    print(f"Experiment {name} stats:\n{json_dumps(stat_name_to_avg_std, indent=2)}\n\n")
    print(f"View experiment & traces at: https://app.parea.ai/experiments/{experiment_uuid}\n")
    save_results_to_dvc_if_init(name, stat_name_to_avg_std)
    return experiment_stats


_experiments = []


@define
class Experiment:
    # If your dataset is defined locally it should be an iterable of k/v
    # pairs matching the expected inputs of your function. To reference a test collection you
    # have saved on Parea, use the collection name as a string.
    data: Union[str, Iterable[dict]]
    # The function to run. This function should accept inputs that match the keys of the data field.
    func: Callable = field()
    experiment_stats: ExperimentStatsSchema = field(init=False, default=None)
    p: Parea = field(default=None)
    name: str = field(init=False)

    def __attrs_post_init__(self):
        global _experiments
        _experiments.append(self)

    def _gen_name_if_none(self, name: Optional[str]):
        if not name:
            self.name = gen_random_name()
            print(f"Experiment name set to: {self.name}, since a name was not provided.")
        else:
            self.name = name

    def run(self, name: Optional[str] = None) -> None:
        """Run the experiment and save the results to DVC.
        param name: The name of the experiment. This name must be unique across experiment runs.
        If no name is provided a memorable name will be generated automatically.
        """
        self._gen_name_if_none(name)
        self.experiment_stats = asyncio.run(experiment(self.name, self.data, self.func, self.p))
