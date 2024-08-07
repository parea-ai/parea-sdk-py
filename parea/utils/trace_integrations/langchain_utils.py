from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, TypedDict, TypeVar, Union

import copy
import datetime
import functools
import json
import logging
import re
import uuid
from enum import Enum

from orjson import orjson

logger = logging.getLogger()


ID_TYPE = Union[uuid.UUID, str]
_MAX_DEPTH = 2


class RunTypeEnum(str, Enum):
    """(Deprecated) Enum for run types. Use string directly."""

    tool = "tool"
    chain = "chain"
    llm = "llm"
    retriever = "retriever"
    embedding = "embedding"
    prompt = "prompt"
    parser = "parser"


class RunLikeDict(TypedDict, total=False):
    """Run-like dictionary, for type-hinting."""

    name: str
    run_type: RunTypeEnum
    start_time: datetime
    inputs: Optional[dict]
    outputs: Optional[dict]
    end_time: Optional[datetime]
    extra: Optional[dict]
    error: Optional[str]
    serialized: Optional[dict]
    parent_run_id: Optional[uuid.UUID]
    manifest_id: Optional[uuid.UUID]
    events: Optional[List[dict]]
    tags: Optional[List[str]]
    inputs_s3_urls: Optional[dict]
    outputs_s3_urls: Optional[dict]
    id: Optional[uuid.UUID]
    session_id: Optional[uuid.UUID]
    session_name: Optional[str]
    reference_example_id: Optional[uuid.UUID]
    input_attachments: Optional[dict]
    output_attachments: Optional[dict]
    trace_id: uuid.UUID
    dotted_order: str


def _as_uuid(value: ID_TYPE, var: Optional[str] = None) -> uuid.UUID:
    try:
        return uuid.UUID(value) if not isinstance(value, uuid.UUID) else value
    except ValueError as e:
        var = var or "value"
        raise Exception(f"{var} must be a valid UUID or UUID string. Got {value}") from e


def _simple_default(obj: Any) -> Any:
    # Don't traverse into nested objects
    try:
        if isinstance(obj, datetime.datetime):
            return obj.isoformat()
        if isinstance(obj, uuid.UUID):
            return str(obj)
        return json.loads(json.dumps(obj))
    except BaseException as e:
        logger.debug(f"Failed to serialize {type(obj)} to JSON: {e}")
        return repr(obj)


def _serialize_json(obj: Any, depth: int = 0, serialize_py: bool = True) -> Any:
    try:
        if depth >= _MAX_DEPTH:
            try:
                return orjson.loads(_dumps_json_single(obj))
            except BaseException:
                return repr(obj)
        if isinstance(obj, bytes):
            return obj.decode("utf-8")
        if isinstance(obj, (set, tuple)):
            return orjson.loads(_dumps_json_single(list(obj)))

        serialization_methods = [
            ("model_dump_json", True),  # Pydantic V2
            ("json", True),  # Pydantic V1
            ("to_json", False),  # dataclass_json
            ("model_dump", True),  # Pydantic V2 with non-serializable fields
            ("dict", False),  # Pydantic V1 with non-serializable fields
        ]
        for attr, exclude_none in serialization_methods:
            if hasattr(obj, attr) and callable(getattr(obj, attr)):
                try:
                    method = getattr(obj, attr)
                    json_str = method(exclude_none=exclude_none) if exclude_none else method()
                    if isinstance(json_str, str):
                        return json.loads(json_str)
                    return orjson.loads(_dumps_json(json_str, depth=depth + 1, serialize_py=serialize_py))
                except Exception as e:
                    logger.debug(f"Failed to serialize {type(obj)} to JSON: {e}")
                    pass
        if serialize_py:
            all_attrs = {}
            if hasattr(obj, "__slots__"):
                all_attrs.update({slot: getattr(obj, slot, None) for slot in obj.__slots__})
            if hasattr(obj, "__dict__"):
                all_attrs.update(vars(obj))
            if all_attrs:
                filtered = {k: v if v is not obj else repr(v) for k, v in all_attrs.items()}
                return orjson.loads(_dumps_json(filtered, depth=depth + 1, serialize_py=serialize_py))
        return repr(obj)
    except BaseException as e:
        logger.debug(f"Failed to serialize {type(obj)} to JSON: {e}")
        return repr(obj)


def _elide_surrogates(s: bytes) -> bytes:
    pattern = re.compile(rb"\\ud[89a-f][0-9a-f]{2}", re.IGNORECASE)
    result = pattern.sub(b"", s)
    return result


def _dumps_json_single(obj: Any, default: Optional[Callable[[Any], Any]] = None) -> bytes:
    try:
        return orjson.dumps(
            obj,
            default=default,
            option=orjson.OPT_SERIALIZE_NUMPY | orjson.OPT_SERIALIZE_DATACLASS | orjson.OPT_SERIALIZE_UUID | orjson.OPT_NON_STR_KEYS,
        )
    except TypeError as e:
        # Usually caused by UTF surrogate characters
        logger.debug(f"Orjson serialization failed: {repr(e)}. Falling back to json.")
        result = json.dumps(
            obj,
            default=_simple_default,
            ensure_ascii=True,
        ).encode("utf-8")
        try:
            result = orjson.dumps(orjson.loads(result.decode("utf-8", errors="surrogateescape")))
        except orjson.JSONDecodeError:
            result = _elide_surrogates(result)
        return result


def _dumps_json(obj: Any, depth: int = 0, serialize_py: bool = True) -> bytes:
    """Serialize an object to a JSON formatted string.
    Parameters
    ----------
    obj : Any
        The object to serialize.
    default : Callable[[Any], Any] or None, default=None
        The default function to use for serialization.
    Returns:
    -------
    str
        The JSON formatted string.
    """
    return _dumps_json_single(obj, functools.partial(_serialize_json, depth=depth, serialize_py=serialize_py))


T = TypeVar("T")


def _middle_copy(val: T, memo: Dict[int, Any], max_depth: int = 4, _depth: int = 0) -> T:
    cls = type(val)

    copier = getattr(cls, "__deepcopy__", None)
    if copier is not None:
        try:
            return copier(memo)
        except BaseException:
            pass
    if _depth >= max_depth:
        return val
    if isinstance(val, dict):
        return {_middle_copy(k, memo, max_depth, _depth + 1): _middle_copy(v, memo, max_depth, _depth + 1) for k, v in val.items()}  # type: ignore[return-value]
    if isinstance(val, list):
        return [_middle_copy(item, memo, max_depth, _depth + 1) for item in val]  # type: ignore[return-value]
    if isinstance(val, tuple):
        return tuple(_middle_copy(item, memo, max_depth, _depth + 1) for item in val)  # type: ignore[return-value]
    if isinstance(val, set):
        return {_middle_copy(item, memo, max_depth, _depth + 1) for item in val}  # type: ignore[return-value]

    return val


def deepish_copy(val: T) -> T:
    """Deep copy a value with a compromise for uncopyable objects.
    Args:
        val: The value to be deep copied.
    Returns:
        The deep copied value.
    """
    memo: Dict[int, Any] = {}
    try:
        return copy.deepcopy(val, memo)
    except BaseException as e:
        # Generators, locks, etc. cannot be copied
        # and raise a TypeError (mentioning pickling, since the dunder methods)
        # are re-used for copying. We'll try to do a compromise and copy
        # what we can
        logger.debug("Failed to deepcopy input: %s", repr(e))
        return _middle_copy(val, memo)
