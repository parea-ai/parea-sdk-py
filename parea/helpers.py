from typing import Any, Optional, Union

import csv
import random
import uuid
from collections.abc import Iterable
from copy import deepcopy
from datetime import datetime

import pytz
from attr import asdict, fields_dict

from parea.constants import ADJECTIVES, NOUNS
from parea.schemas.models import Completion, TraceLog, UpdateLog
from parea.utils.universal_encoder import json_dumps


def gen_trace_id() -> str:
    """Generate a unique trace id for each chain of requests"""
    return str(uuid.uuid4())


def write_trace_logs_to_csv(path_csv: str, trace_logs: list[TraceLog]):
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


def calculate_avg_as_string(values: list[Optional[float]]) -> str:
    if not values:
        return "N/A"
    values = [x for x in values if x is not None]
    avg = sum(values) / len(values)
    return f"{avg:.2f}"


def duplicate_dicts(data: Iterable[dict], n: int) -> Iterable[dict]:
    return [deepcopy(item) for item in data for _ in range(n)]


def serialize_metadata_values(log_data: Union[TraceLog, UpdateLog, Completion]) -> Union[TraceLog, UpdateLog, Completion]:
    def serialize_values(metadata: dict[str, Any]) -> dict[str, str]:
        return {k: json_dumps(v) for k, v in metadata.items()}

    if isinstance(log_data, UpdateLog) and log_data.field_name_to_value_map:
        if "metadata" in log_data.field_name_to_value_map:
            serialized_values = serialize_values(log_data.field_name_to_value_map["metadata"])
            log_data.field_name_to_value_map["metadata"] = serialized_values
    elif log_data.metadata:
        serialized_values = serialize_values(log_data.metadata)
        log_data.metadata = serialized_values

    return log_data


def timezone_aware_now() -> datetime:
    return datetime.now(pytz.utc)
