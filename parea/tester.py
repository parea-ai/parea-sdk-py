from typing import List

import argparse
import concurrent
import csv
import importlib
import os
import sys
import time
from importlib import util

from attr import asdict, fields_dict
from tqdm import tqdm

from parea.cache.redis import RedisLRUCache
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


def read_input_file(file_path):
    with open(file_path) as file:
        reader = csv.reader(file)
        inputs = list(reader)
    return inputs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--user_func", help="User function to test e.g., path/to/user_code.py:argument_chain", type=str)
    parser.add_argument("--inputs", help="Path to the input CSV file", type=str)
    args = parser.parse_args()

    fn = load_from_path(*args.user_func.rsplit(":", 1))

    data_inputs = read_input_file(args.inputs)

    redis_logs_key = f"parea-trace-logs-{int(time.time())}"
    os.putenv("_redis_logs_key", redis_logs_key)

    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [executor.submit(fn, data_input) for data_input in data_inputs]
        for f in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            pass
        print(f"Done with {len(futures)} inputs")

        redis_cache = RedisLRUCache(key_logs=redis_logs_key)

        trace_logs: list[TraceLog] = redis_cache.read_logs()

        # write to csv
        path_csv = f"trace_logs-{int(time.time())}.csv"
        with open(path_csv, "w", newline="") as file:
            # write header
            columns = fields_dict(TraceLog).keys()
            writer = csv.DictWriter(file, fieldnames=columns)
            writer.writeheader()
            # write rows
            for trace_log in trace_logs:
                writer.writerow(asdict(trace_log))

        print(f"Wrote CSV of results to: {path_csv}")
