# type: ignore[attr-defined]
# flake8: noqa

"""
    Parea API SDK

    The Parea SDK allows you to interact with Parea from your product or service.
    To install the official [Python SDK](https://pypi.org/project/parea/),
    run the following command:  ```bash pip install parea ```.
"""
import sys

from parea.api_client import get_version
from parea.cache import InMemoryCache
from parea.client import Parea
from parea.experiment.cli import experiment as _experiment_cli
from parea.experiment.dvc import parea_dvc_initialized
from parea.experiment.experiment import Experiment
from parea.helpers import gen_trace_id, write_trace_logs_to_csv
from parea.parea_logger import parea_logger
from parea.utils.trace_utils import clear_trace_context, get_current_trace_id, get_root_trace_id, trace, trace_insert
from parea.wrapper.openai_raw_api_tracer import aprocess_stream_and_yield, process_stream_and_yield
from parea.wrapper.utils import convert_openai_raw_to_log

version: str = get_version()


def main():
    args = sys.argv[1:]
    if args[0] == "experiment":
        _experiment_cli(args[1:])
    elif args[0] == "dvc-init":
        parea_dvc_initialized(only_check=False)
    else:
        print(f"Unknown command: '{args[0]}'")


if __name__ == "__main__":
    main()
