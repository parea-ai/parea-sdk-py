from typing import List

import argparse
import csv
import os
import sys
import traceback
from importlib import util

from .experiment import _experiments


def load_from_path(module_path):
    # Ensure the directory of user-provided script is in the system path
    dir_name = os.path.dirname(module_path)
    if dir_name not in sys.path:
        sys.path.insert(0, dir_name)

    module_name = os.path.basename(module_path)
    spec = util.spec_from_file_location(module_name, module_path)
    module = util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if spec.name not in sys.modules:
        sys.modules[spec.name] = module


def read_input_file(file_path) -> List[dict]:
    with open(file_path) as file:
        reader = csv.DictReader(file)
        inputs = list(reader)
    return inputs


def experiment(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("file", help="Path to the experiment", type=str)
    parser.add_argument("--run_name", help="Name of the experiment run", type=str, default=None)

    parsed_args = parser.parse_args(args)

    try:
        load_from_path(parsed_args.file)
    except Exception as e:
        print(f"Error loading function: {e}\n", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)

    for _experiment in _experiments:
        _experiment.run(parsed_args.run_name)
