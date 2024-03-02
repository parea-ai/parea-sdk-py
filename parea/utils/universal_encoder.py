from typing import Any

import dataclasses
import datetime
import json
import logging
from decimal import Decimal
from enum import Enum
from uuid import UUID

import attrs
from pydantic import BaseModel

from parea.types import OpenAIAsyncStreamWrapper, OpenAIStreamWrapper

logger = logging.getLogger()


def is_dataclass_instance(obj):
    return dataclasses.is_dataclass(obj) and not isinstance(obj, type)


def is_attrs_instance(obj):
    return attrs.has(obj)


def is_numpy_instance(obj):
    try:
        import numpy as np
    except ImportError:
        np = None

    return np and isinstance(obj, np.ndarray)


def is_pandas_instance(obj):
    try:
        import pandas as pd
    except ImportError:
        pd = None

    return pd and isinstance(obj, pd.DataFrame)


class UniversalEncoder(json.JSONEncoder):
    """
    A JSON encoder that can handle additional types such as dataclasses, attrs, and more.
    """

    def default(self, obj: Any):
        if isinstance(obj, str):
            return obj
        elif isinstance(obj, set):
            return list(obj)
        elif isinstance(obj, Enum):
            try:
                return str(obj.value)
            except Exception:
                return str(obj)
        elif isinstance(obj, (OpenAIStreamWrapper, OpenAIAsyncStreamWrapper)):
            return str(obj)
        elif is_dataclass_instance(obj):
            return dataclasses.asdict(obj)
        elif is_attrs_instance(obj):
            return attrs.asdict(obj)
        elif isinstance(obj, BaseModel):
            try:
                return obj.model_dump()
            except Exception:
                return obj.dict()
        elif isinstance(obj, (datetime.datetime, datetime.date, datetime.time)):
            return obj.isoformat()
        elif isinstance(obj, datetime.timedelta):
            return obj.total_seconds()
        elif isinstance(obj, UUID):
            return str(obj)
        elif isinstance(obj, Decimal):
            return float(obj)
        elif callable(obj):
            return f"<callable {obj.__name__}>"
        elif isinstance(obj, bytes):
            return obj.decode(errors="ignore")
        elif is_numpy_instance(obj):
            return obj.tolist()
        elif is_pandas_instance(obj):
            return obj.to_dict(orient="records")
        else:
            return super().default(obj)


def json_dumps(obj, **kwargs) -> str:
    return json.dumps(obj, cls=UniversalEncoder, **kwargs) if not isinstance(obj, str) else obj
