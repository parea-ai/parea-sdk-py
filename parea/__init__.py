# type: ignore[attr-defined]
# flake8: noqa

"""
    Parea API SDK

    The Parea SDK allows you to interact with Parea from your product or service.
    To install the official [Python SDK](https://pypi.org/project/parea/),
    run the following command:  ```bash pip install parea ```.
"""
import sys
from importlib import metadata as importlib_metadata

from parea.cache import InMemoryCache, RedisCache
from parea.client import Parea
from parea.experiment.cli import experiment as _experiment_cli
from parea.experiment.experiment import Experiment
from parea.utils.trace_utils import trace


def get_version() -> str:
    try:
        return importlib_metadata.version(__name__)
    except importlib_metadata.PackageNotFoundError:  # pragma: no cover
        return "unknown"


version: str = get_version()


def main():
    args = sys.argv[1:]
    if args[0] == "experiment":
        _experiment_cli(args[1:])
    else:
        print(f"Unknown command: '{args[0]}'")


if __name__ == "__main__":
    main()
