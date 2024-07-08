from typing import Any

import dataclasses
import datetime
import json
import logging
from decimal import Decimal
from enum import Enum
from uuid import UUID

import attrs
import openai
from pydantic import BaseModel

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


def is_openai_stream_wrapper(obj):
    if openai.__version__.startswith("0."):
        return False
    else:
        from parea.types import OpenAIAsyncStreamWrapper, OpenAIStreamWrapper

        return isinstance(obj, (OpenAIStreamWrapper, OpenAIAsyncStreamWrapper))


class UniversalEncoder(json.JSONEncoder):
    """
    A JSON encoder that can handle additional types such as dataclasses, attrs, and more.
    """

    def handle_dspy_response(self, obj) -> Any:
        try:
            import dspy
        except ImportError:
            return None

        from dsp.templates.template_v3 import Template
        from dspy.primitives.example import Example

        if hasattr(obj, "completions") and hasattr(obj.completions, "_completions"):
            # multiple completions
            return obj.completions._completions
        elif hasattr(obj, "_asdict"):
            # convert namedtuples to dictionaries
            return obj._asdict()
        elif isinstance(obj, Example):
            # handles Prediction objects and other sub-classes of Example
            return getattr(obj, "_store", {})
        elif isinstance(obj, Template):
            return {
                "fields": [self.default(field) for field in obj.fields],
                "instructions": obj.instructions,
            }
        else:
            return None

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
        elif is_openai_stream_wrapper(obj):
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
        elif is_numpy_instance(obj):
            return obj.tolist()
        elif is_pandas_instance(obj):
            return obj.to_dict(orient="records")
        elif dspy_response := self.handle_dspy_response(obj):
            return dspy_response
        elif callable(obj):
            try:
                return f"<callable {obj.__name__}>"
            except AttributeError:
                return str(obj)
        elif isinstance(obj, bytes):
            return obj.decode(errors="ignore")
        else:
            return super().default(obj)


def json_dumps(obj, **kwargs) -> str:
    try:
        return json.dumps(obj, cls=UniversalEncoder, **kwargs) if not isinstance(obj, str) else obj
    except TypeError as e:
        logger.debug(f"Error serializing object: {obj} with error: {e}")
        return str(obj)
