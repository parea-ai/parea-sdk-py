import concurrent
import importlib
import os
import csv
import argparse
import sys
from importlib import machinery, util


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

    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = list(executor.map(fn, data_inputs))

    for i, result in enumerate(results):
        print(f'input: {data_inputs[i]}')
        print(f'result: {result}')
        print()
