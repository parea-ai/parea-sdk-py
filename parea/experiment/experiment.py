from typing import Callable, Dict, Iterable, List

import asyncio
import inspect
import json
import os

from attrs import define, field
from dotenv import load_dotenv
from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio

from parea.client import Parea
from parea.constants import PAREA_OS_ENV_EXPERIMENT_UUID
from parea.schemas.models import CreateExperimentRequest, ExperimentSchema, ExperimentStatsSchema, TraceStatsSchema
from parea.utils.trace_utils import thread_ids_running_evals


def calculate_avg_as_string(values: List[float]) -> str:
    if not values:
        return "N/A"
    values = [x for x in values if x is not None]
    avg = sum(values) / len(values)
    return f"{avg:.2f}"


def calculate_avg_std_for_experiment(experiment_stats: ExperimentStatsSchema) -> Dict[str, str]:
    trace_stats: List[TraceStatsSchema] = experiment_stats.parent_trace_stats
    latency_values = [trace_stat.latency for trace_stat in trace_stats]
    input_tokens_values = [trace_stat.input_tokens for trace_stat in trace_stats]
    output_tokens_values = [trace_stat.output_tokens for trace_stat in trace_stats]
    total_tokens_values = [trace_stat.total_tokens for trace_stat in trace_stats]
    cost_values = [trace_stat.cost for trace_stat in trace_stats]
    score_name_to_values: Dict[str, List[float]] = {}
    for trace_stat in trace_stats:
        if trace_stat.scores:
            for score in trace_stat.scores:
                score_name_to_values[score.name] = score_name_to_values.get(score.name, []) + [score.score]

    return {
        "latency": calculate_avg_as_string(latency_values),
        "input_tokens": calculate_avg_as_string(input_tokens_values),
        "output_tokens": calculate_avg_as_string(output_tokens_values),
        "total_tokens": calculate_avg_as_string(total_tokens_values),
        "cost": calculate_avg_as_string(cost_values),
        **{score_name: calculate_avg_as_string(score_values) for score_name, score_values in score_name_to_values.items()},
    }


def async_wrapper(fn, **kwargs):
    return asyncio.run(fn(**kwargs))


async def experiment(name: str, data: Iterable[Dict], func: Callable) -> ExperimentStatsSchema:
    """Creates an experiment and runs the function on the data iterator."""
    load_dotenv()

    if not (parea_api_key := os.getenv("PAREA_API_KEY")):
        raise ValueError("Please set the PAREA_API_KEY environment variable")
    p = Parea(api_key=parea_api_key)

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

    experiment_stats: ExperimentStatsSchema = p.finish_experiment(experiment_uuid)
    stat_name_to_avg_std = calculate_avg_std_for_experiment(experiment_stats)
    print(f"Experiment stats:\n{json.dumps(stat_name_to_avg_std, indent=2)}\n\n")
    print(f"View experiment & its traces at: https://app.parea.ai/experiments/{experiment_uuid}\n")
    return experiment_stats


_experiments = []


@define
class Experiment:
    name: str = field(init=True)
    data: Iterable[Dict] = field(init=True)
    func: Callable = field(init=True)
    experiment_stats: ExperimentStatsSchema = field(init=False, default=None)

    def __attrs_post_init__(self):
        global _experiments
        _experiments.append(self)

    def run(self):
        self.experiment_stats = asyncio.run(experiment(self.name, self.data, self.func))
