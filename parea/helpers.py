import csv
import time
import uuid

from attr import asdict, fields_dict

from parea.schemas.models import TraceLog


def gen_trace_id() -> str:
    """Generate a unique trace id for each chain of requests"""
    return str(uuid.uuid4())


def to_date_and_time_string(timestamp: float) -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S %Z", time.localtime(timestamp))


def date_and_time_string_to_timestamp(date_and_time_string: str) -> float:
    return time.mktime(time.strptime(date_and_time_string, "%Y-%m-%d %H:%M:%S %Z"))


def write_trace_logs_to_csv(path_csv: str, trace_logs: list[TraceLog]):
    with open(path_csv, "w", newline="") as file:
        # write header
        columns = fields_dict(TraceLog).keys()
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        # write rows
        for trace_log in trace_logs:
            writer.writerow(asdict(trace_log))
