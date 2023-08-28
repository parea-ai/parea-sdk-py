import functools
import inspect
from typing import Any, Callable, List
import time

from parea.schemas.models import TraceLog
from parea.utils.trace_utils import to_date_and_time_string


class Wrapper:
    def __init__(self, module: Any, func_names: List[str], resolver: Callable, log: Callable) -> None:
        self.resolver = resolver
        self.log = log
        self.set_funcs(module, func_names)

    def set_funcs(self, module: Any, func_names: List[str]):
        for func_name in func_names:
            func_name_parts = func_name.split('.')
            original = functools.reduce(getattr, func_name_parts, module)
            setattr(
                self.get_func_module(module, func_name_parts),
                func_name_parts[-1],
                self._wrapped_func(original)
            )

    @staticmethod
    def get_func_module(module: Any, func_name_parts: List[str]):
        return module if len(func_name_parts) == 1 else functools.reduce(getattr, func_name_parts[:-1], module)

    def _wrapped_func(self, original_func: Callable) -> Callable:
        unwrapped_func = self.get_unwrapped_func(original_func)
        return self.get_wrapper(unwrapped_func, original_func)

    @staticmethod
    def get_unwrapped_func(original_func: Callable) -> Callable:
        while hasattr(original_func, '__wrapped__'):
            original_func = original_func.__wrapped__
        return original_func

    def get_wrapper(self, unwrapped_func: Callable, original_func: Callable):
        if inspect.iscoroutinefunction(unwrapped_func):
            return self.async_wrapper(original_func)
        else:
            return self.sync_wrapper(original_func)

    def async_wrapper(self, orig_func: Callable) -> Callable:
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            response = None
            error = None
            try:
                response = await orig_func(*args, **kwargs)
                return response
            except Exception as e:
                error = e
                raise
            finally:
                self.log_result(start_time, response, error, args, kwargs)
        return wrapper

    def sync_wrapper(self, orig_func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            start_time = time.time()
            response = None
            error = None
            try:
                response = orig_func(*args, **kwargs)
                return response
            except Exception as e:
                error = e
                raise
            finally:
                self.log_result(start_time, response, error, args, kwargs)
        return wrapper

    def log_result(self, start_time: float, response: Any, error: Any, args, kwargs):
        end_time = time.time()
        latency = end_time - start_time

        start_timestamp = to_date_and_time_string(start_time)
        end_timestamp = to_date_and_time_string(end_time)

        log_data_dict = {
            'trace_id': '',
            'start_timestamp': start_timestamp,
            'end_timestamp': end_timestamp,
            "error": str(error),
            "status": "success" if not error else "error",
            "latency": latency,
        }

        log_data_dict_provider = self.resolver(args, kwargs, response)
        self.log(TraceLog(**log_data_dict, **log_data_dict_provider))

