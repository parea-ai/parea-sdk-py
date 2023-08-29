from typing import Any, Callable, List, Tuple

import functools
import inspect
import time
from uuid import uuid4

from parea.schemas.models import TraceLog
from parea.utils.trace_utils import default_logger, to_date_and_time_string, trace_context, trace_data


class Wrapper:
    def __init__(self, module: Any, func_names: List[str], resolver: Callable, log: Callable = default_logger) -> None:
        self.resolver = resolver
        self.log = log
        self.wrap_functions(module, func_names)

    def wrap_functions(self, module: Any, func_names: List[str]):
        for func_name in func_names:
            func_name_parts = func_name.split(".")
            original = functools.reduce(getattr, func_name_parts, module)
            setattr(module if len(func_name_parts) == 1 else functools.reduce(getattr, func_name_parts[:-1], module), func_name_parts[-1], self._wrapped_func(original))

    def _wrapped_func(self, original_func: Callable) -> Callable:
        unwrapped_func = original_func
        while hasattr(original_func, "__wrapped__"):
            unwrapped_func = original_func.__wrapped__
        return self._get_decorator(unwrapped_func, original_func)

    def _get_decorator(self, unwrapped_func: Callable, original_func: Callable):
        if inspect.iscoroutinefunction(unwrapped_func):
            return self.async_decorator(original_func)
        else:
            return self.sync_decorator(original_func)

    def _init_trace(self) -> Tuple[str, float]:
        start_time = time.time()
        trace_id = str(uuid4())
        trace_context.get().append(trace_id)

        trace_data.get()[trace_id] = TraceLog(
            trace_id=trace_id,
            start_timestamp=to_date_and_time_string(start_time),
            trace_name="LLM",
            end_user_identifier=None,
            metadata=None,
            target=None,
            tags=None,
            inputs={},
        )

        parent_trace_id = trace_context.get()[-2] if len(trace_context.get()) > 1 else None
        if not parent_trace_id:
            # we don't have a parent trace id, so we need to create one
            parent_trace_id = str(uuid4())
            trace_context.get().insert(0, parent_trace_id)
            trace_data.get()[parent_trace_id] = TraceLog(
                trace_id=parent_trace_id,
                start_timestamp=to_date_and_time_string(start_time),
                end_user_identifier=None,
                metadata=None,
                target=None,
                tags=None,
                inputs={},
            )
        trace_data.get()[parent_trace_id].children.append(trace_id)
        self.log(parent_trace_id)

        return trace_id, start_time

    def async_decorator(self, orig_func: Callable) -> Callable:
        async def wrapper(*args, **kwargs):
            trace_id, start_time = self._init_trace()
            response = None
            error = None
            try:
                response = await orig_func(*args, **kwargs)
                return response
            except Exception as e:
                error = str(e)
                raise
            finally:
                self._cleanup_trace(trace_id, start_time, error, response, args, kwargs)

        return wrapper

    def sync_decorator(self, orig_func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            trace_id, start_time = self._init_trace()
            response = None
            error = None
            try:
                response = orig_func(*args, **kwargs)
                return response
            except Exception as e:
                error = str(e)
                raise
            finally:
                self._cleanup_trace(trace_id, start_time, error, response, args, kwargs)

        return wrapper

    def _cleanup_trace(self, trace_id: str, start_time: float, error: str, response: Any, args, kwargs):
        end_time = time.time()
        trace_data.get()[trace_id].end_timestamp = to_date_and_time_string(end_time)
        trace_data.get()[trace_id].latency = end_time - start_time

        if error:
            trace_data.get()[trace_id].error = error
            trace_data.get()[trace_id].status = "error"
        else:
            trace_data.get()[trace_id].status = "success"

        self.resolver(trace_id, args, kwargs, response)

        self.log(trace_id)
        trace_context.get().pop()
