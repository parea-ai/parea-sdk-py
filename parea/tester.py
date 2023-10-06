import concurrent
import os
import csv
import argparse
import sys
from importlib import machinery, util


def load_from_path(path_to_module, attr_name):
    module_name = os.path.basename(path_to_module)
    loader = machinery.SourceFileLoader(module_name, path_to_module)
    spec = util.spec_from_file_location(module_name, path_to_module, loader=loader)
    module = util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    fn = getattr(module, attr_name)
    return fn


def read_input_file(file_path):
    with open(file_path, 'r') as file:
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

    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(executor.map(fn, data_inputs))

    for i, result in enumerate(results):
        print(f'input: {data_inputs[i]}')
        print(f'result: {result}')
        print()
