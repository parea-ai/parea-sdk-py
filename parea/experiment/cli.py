import argparse
import csv
import importlib
import os
import sys
import traceback
from importlib import util

from .experiment import experiment as experiment_orig


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


def experiment(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", help="Name of the experiment", type=str, required=True)
    parser.add_argument("--func", help="Function to test e.g., path/to/my_code.py:argument_chain", type=str, required=True)
    parser.add_argument("--csv_path", help="Path to the input CSV file", type=str, required=True)
    parsed_args = parser.parse_args(args)

    try:
        func = load_from_path(*parsed_args.func.rsplit(":", 1))
    except Exception as e:
        print(f"Error loading function: {e}\n", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)
    try:
        data = read_input_file(parsed_args.csv_path)
    except Exception as e:
        print(f"Error reading input file: {e}\n", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)

    experiment_orig(name=parsed_args.name, func=func, data=data)
