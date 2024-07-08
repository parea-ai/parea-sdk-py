from typing import Any, Dict, Iterable, List, Optional, Union

import csv
import random
import uuid
from copy import deepcopy
from datetime import datetime

import pytz
from attr import asdict, fields_dict
from cattrs import GenConverter

from parea.constants import ADJECTIVES, NOUNS, TURN_OFF_PAREA_LOGGING
from parea.schemas.models import Completion, TraceLog, TraceLogTree, UpdateLog
from parea.utils.universal_encoder import json_dumps


def gen_trace_id() -> str:
    """Generate a unique trace id for each chain of requests"""
    return str(uuid.uuid4())


def write_trace_logs_to_csv(path_csv: str, trace_logs: List[TraceLog]):
    with open(path_csv, "w", newline="") as file:
        # write header
        columns = fields_dict(TraceLog).keys()
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        # write rows
        for trace_log in trace_logs:
            writer.writerow(asdict(trace_log))


def gen_random_name():
    random_generator = random.Random()
    adjective = random_generator.choice(ADJECTIVES)
    noun = random_generator.choice(NOUNS)
    return f"{adjective}-{noun}"


def calculate_avg_as_string(values: List[Optional[float]]) -> str:
    if not values:
        return "N/A"
    values = [x for x in values if x is not None]
    avg = sum(values) / len(values)
    return f"{avg:.2f}"


def duplicate_dicts(data: Iterable[dict], n: int) -> Iterable[dict]:
    return [deepcopy(item) for item in data for _ in range(n)]


def serialize_metadata_values(log_data: Union[TraceLog, UpdateLog, Completion]) -> Union[TraceLog, UpdateLog, Completion]:
    def serialize_values(metadata: Dict[str, Any]) -> Dict[str, str]:
        return {k: json_dumps(v) for k, v in metadata.items()}

    if isinstance(log_data, UpdateLog) and log_data.field_name_to_value_map:
        if "metadata" in log_data.field_name_to_value_map:
            serialized_values = serialize_values(log_data.field_name_to_value_map["metadata"])
            log_data.field_name_to_value_map["metadata"] = serialized_values
        return log_data

    if log_data.metadata:
        serialized_values = serialize_values(log_data.metadata)
        log_data.metadata = serialized_values

    # Support openai vision content format
    messages = []
    if isinstance(log_data, TraceLog) and log_data.configuration:
        messages = log_data.configuration.messages or []
    elif isinstance(log_data, Completion) and log_data.llm_configuration:
        messages = log_data.llm_configuration.messages or []
    for message in messages:
        if isinstance(message, dict) and "content" in message and not isinstance(message["content"], str):
            message["content"] = json_dumps(message["content"])

    return log_data


def timezone_aware_now() -> datetime:
    return datetime.now(pytz.utc)


def structure_trace_log_from_api(d: dict) -> TraceLogTree:
    def structure_union_type(obj: Any, cl: type) -> Any:
        if isinstance(obj, str):
            return obj
        elif isinstance(obj, dict):
            return obj
        else:
            return None

    def structure_float_or_none(obj: Any, cl: type) -> Optional[float]:
        if obj is None:
            return None
        try:
            return float(obj)
        except (ValueError, TypeError):
            return None

    converter = GenConverter()
    converter.register_structure_hook(Union[str, Dict[str, str], None], structure_union_type)
    converter.register_structure_hook(float, structure_float_or_none)
    converter.register_structure_hook(Optional[float], structure_float_or_none)
    return converter.structure(d, TraceLogTree)


def structure_trace_logs_from_api(data: List[dict]) -> List[TraceLogTree]:
    return [structure_trace_log_from_api(d) for d in data]


PAREA_LOGGING_DISABLED = False


def disable_parea_logging():
    global PAREA_LOGGING_DISABLED
    PAREA_LOGGING_DISABLED = True


def enable_logging():
    global PAREA_LOGGING_DISABLED
    PAREA_LOGGING_DISABLED = False


def is_logging_disabled() -> bool:
    global PAREA_LOGGING_DISABLED
    return PAREA_LOGGING_DISABLED or TURN_OFF_PAREA_LOGGING


class TurnOffPareaLogging:
    def __enter__(self):
        disable_parea_logging()

    def __exit__(self, exc_type, exc_val, exc_tb):
        enable_logging()
