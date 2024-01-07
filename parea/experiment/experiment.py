import asyncio
import concurrent
import inspect
import json
import os
import time
from math import sqrt
from typing import Dict, List, Callable, Iterator

from dotenv import load_dotenv
from tqdm import tqdm

from parea.client import Parea
from parea.constants import PAREA_OS_ENV_EXPERIMENT_UUID
from parea.schemas.models import CreateExperimentRequest, ExperimentSchema, ExperimentStatsSchema, TraceStatsSchema


def calculate_avg_std_as_string(values: List[float]) -> str:
    if not values:
        return "N/A"
    values = [x for x in values if x is not None]
    avg = sum(values) / len(values)
    std = sqrt(sum((x - avg) ** 2 for x in values) / len(values))
    return f"{avg:.2f} ± {std:.2f}"


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
        "latency": calculate_avg_std_as_string(latency_values),
        "input_tokens": calculate_avg_std_as_string(input_tokens_values),
        "output_tokens": calculate_avg_std_as_string(output_tokens_values),
        "total_tokens": calculate_avg_std_as_string(total_tokens_values),
        "cost": calculate_avg_std_as_string(cost_values),
        **{score_name: calculate_avg_std_as_string(score_values) for score_name, score_values in score_name_to_values.items()}
    }


def async_wrapper(fn, **kwargs):
    return asyncio.run(fn(**kwargs))


def experiment(name: str, data: Iterator, func: Callable):
    load_dotenv()

    if not (parea_api_key := os.getenv("PAREA_API_KEY")):
        raise ValueError("Please set the PAREA_API_KEY environment variable")
    p = Parea(api_key=parea_api_key)

    experiment_schema: ExperimentSchema = p.create_experiment(CreateExperimentRequest(name=name))
    experiment_uuid = experiment_schema.uuid
    os.putenv(PAREA_OS_ENV_EXPERIMENT_UUID, experiment_uuid)

    with concurrent.futures.ProcessPoolExecutor(max_workers=os.cpu_count() or 1) as executor:
        if inspect.iscoroutinefunction(func):
            futures = [executor.submit(async_wrapper, func, **data_input) for data_input in data]
        else:
            futures = [executor.submit(func, **data_input) for data_input in data]
        for _ in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            pass

        time.sleep(5)  # wait for all trace logs to be written to DB
        experiment_stats: ExperimentStatsSchema = p.get_experiment_stats(experiment_uuid)
        stat_name_to_avg_std = calculate_avg_std_for_experiment(experiment_stats)
        print(f"Experiment stats:\n{json.dumps(stat_name_to_avg_std, indent=2)}\n\n")
        print(f"View experiment & its traces at: https://app.parea.ai/experiments/{experiment_uuid}\n")
