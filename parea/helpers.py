import time
import uuid


def gen_trace_id() -> str:
    """Generate a unique trace id for each chain of requests"""
    return str(uuid.uuid4())


def to_date_and_time_string(timestamp: float) -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S %Z", time.localtime(timestamp))


def date_and_time_string_to_timestamp(date_and_time_string: str) -> float:
    return time.mktime(time.strptime(date_and_time_string, "%Y-%m-%d %H:%M:%S %Z"))
