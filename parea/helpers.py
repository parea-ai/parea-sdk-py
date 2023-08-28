import uuid
from datetime import datetime


def gen_trace_id() -> str:
    """Generate a unique trace id for each chain of requests"""
    return str(uuid.uuid4())


def to_date_and_time_string(timestamp: float) -> str:
    """Convert a timestamp to a date and time string"""
    return datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S %Z").strip()
