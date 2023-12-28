import argparse
import asyncio
import concurrent
import csv
import importlib
import inspect
import json
import os
import sys
import time
from importlib import util
from math import sqrt
from typing import Dict, List

from tqdm import tqdm

from parea import Parea
from parea.cache.redis import RedisCache
from parea.constants import PAREA_OS_ENV_EXPERIMENT_UUID
from parea.helpers import write_trace_logs_to_csv
from parea.schemas.models import TraceLog, CreateExperimentRequest, Experiment, ExperimentStatsSchema, TraceStatsSchema


def load_from_path(module_path, attr_name):
    # Ensure the directory of user-provided script is in the system path
    dir_name = os.path.dirname(module_path)
    if dir_name not in sys.path:
        sys.path.insert(0, dir_name)

    module_name = os.path.basename(module_path)
    # Add .py extension back in to allow import correctly
    module_path_with_ext = f"{module_path}.py"

    spec = importlib.util.spec_from_file_location(module_name, module_path_with_ext)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if spec.name not in sys.modules:
        sys.modules[spec.name] = module

    fn = getattr(module, attr_name)
    return fn


def read_input_file(file_path) -> list[dict]:
    with open(file_path) as file:
        reader = csv.DictReader(file)
        inputs = list(reader)
    return inputs


def async_wrapper(fn, **kwargs):
    return asyncio.run(fn(**kwargs))


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


def run_benchmark(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", help="Name of the experiment", type=str, required=True)
    parser.add_argument("--func", help="Function to test e.g., path/to/my_code.py:argument_chain", type=str, required=True)
    parser.add_argument("--csv_path", help="Path to the input CSV file", type=str, required=True)
    parser.add_argument("--redis_host", help="Redis host", type=str, default=None)
    parser.add_argument("--redis_port", help="Redis port", type=int, default=None)
    parser.add_argument("--redis_password", help="Redis password", type=str, default=None)
    parsed_args = parser.parse_args(args)

    fn = load_from_path(*parsed_args.func.rsplit(":", 1))

    data_inputs = read_input_file(parsed_args.csv_path)

    if parea_api_key := os.getenv("PAREA_API_KEY") is None:
        raise ValueError("Please set the PAREA_API_KEY environment variable")
    p = Parea(api_key=parea_api_key)

    experiment: Experiment = p.create_experiment(CreateExperimentRequest(name=parsed_args.name))
    os.putenv(PAREA_OS_ENV_EXPERIMENT_UUID, experiment.uuid)

    if is_using_redis := parsed_args.redis_host and parsed_args.redis_port:
        redis_logs_key = f"parea-trace-logs-{int(time.time())}"
        os.putenv("_parea_redis_logs_key", redis_logs_key)

    with concurrent.futures.ProcessPoolExecutor() as executor:
        if inspect.iscoroutinefunction(fn):
            futures = [executor.submit(async_wrapper, fn, **data_input) for data_input in data_inputs]
        else:
            futures = [executor.submit(fn, **data_input) for data_input in data_inputs]
        for _ in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            pass

        experiment_stats: ExperimentStatsSchema = p.get_experiment_stats(experiment.uuid)
        stat_name_to_avg_std = calculate_avg_std_for_experiment(experiment_stats)
        print(f"Experiment stats:\n{json.dumps(stat_name_to_avg_std, indent=2)}")

        if is_using_redis:
            redis_cache = RedisCache(key_logs=redis_logs_key, host=args.redis_host, port=args.redis_port, password=args.redis_password)
            # write to csv
            path_csv = f"trace_logs-{int(time.time())}.csv"
            trace_logs: list[TraceLog] = redis_cache.read_logs()
            write_trace_logs_to_csv(path_csv, trace_logs)
            print(f"Wrote CSV of results to: {path_csv}")
