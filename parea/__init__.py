# type: ignore[attr-defined]
# flake8: noqa

"""
    Parea API SDK

    The Parea SDK allows you to interact with Parea from your product or service.
    To install the official [Python SDK](https://pypi.org/project/parea/),
    run the following command:  ```bash pip install parea ```.
"""

from importlib import metadata as importlib_metadata

import parea.wrapper  # noqa: F401
from parea.client import Parea


def get_version() -> str:
    try:
        return importlib_metadata.version(__name__)
    except importlib_metadata.PackageNotFoundError:  # pragma: no cover
        return "unknown"


version: str = get_version()
