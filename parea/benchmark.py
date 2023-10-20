import argparse
import asyncio
import concurrent
import csv
import importlib
import inspect
import os
import sys
import time
from importlib import util

from attr import asdict, fields_dict
from tqdm import tqdm

from parea.cache.redis import RedisCache
from parea.helpers import write_trace_logs_to_csv
from parea.schemas.models import TraceLog


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


def run_benchmark(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("--func", help="Function to test e.g., path/to/my_code.py:argument_chain", type=str, required=True)
    parser.add_argument("--csv_path", help="Path to the input CSV file", type=str, required=True)
    parser.add_argument("--redis_host", help="Redis host", type=str, default=os.getenv("REDIS_HOST", "localhost"))
    parser.add_argument("--redis_port", help="Redis port", type=int, default=int(os.getenv("REDIS_PORT", 6379)))
    parser.add_argument("--redis_password", help="Redis password", type=str, default=None)
    parsed_args = parser.parse_args(args)

    fn = load_from_path(*parsed_args.func.rsplit(":", 1))

    data_inputs = read_input_file(parsed_args.csv_path)

    redis_logs_key = f"parea-trace-logs-{int(time.time())}"
    os.putenv("_parea_redis_logs_key", redis_logs_key)

    with concurrent.futures.ProcessPoolExecutor() as executor:
        if inspect.iscoroutinefunction(fn):
            futures = [executor.submit(async_wrapper, fn, **data_input) for data_input in data_inputs]
        else:
            futures = [executor.submit(fn, **data_input) for data_input in data_inputs]
        for _ in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            pass
        print(f"Done with {len(futures)} inputs")

        redis_cache = RedisCache(key_logs=redis_logs_key, host=args.redis_host, port=args.redis_port, password=args.redis_password)

        # write to csv
        path_csv = f"trace_logs-{int(time.time())}.csv"
        trace_logs: list[TraceLog] = redis_cache.read_logs()
        write_trace_logs_to_csv(path_csv, trace_logs)
        print(f"Wrote CSV of results to: {path_csv}")
