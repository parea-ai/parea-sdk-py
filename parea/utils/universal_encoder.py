from typing import Any

import dataclasses
import datetime
import json
from uuid import UUID

import attrs


def is_dataclass_instance(obj):
    return dataclasses.is_dataclass(obj) and not isinstance(obj, type)


def is_attrs_instance(obj):
    return attrs.has(obj)


class UniversalEncoder(json.JSONEncoder):
    def default(self, obj: Any):
        if isinstance(obj, str):
            return obj
        elif isinstance(obj, set):
            return list(obj)
        elif is_dataclass_instance(obj):
            return dataclasses.asdict(obj)
        elif is_attrs_instance(obj):
            return attrs.asdict(obj)
        elif isinstance(obj, (datetime.datetime, datetime.date, datetime.time)):
            return obj.isoformat()
        elif isinstance(obj, datetime.timedelta):
            return obj.total_seconds()
        elif isinstance(obj, UUID):
            return str(obj)
        else:
            return super().default(obj)


def json_dumps(obj, **kwargs):
    return json.dumps(obj, cls=UniversalEncoder, **kwargs)
