from typing import Any, Callable

import functools
import inspect
import logging
import os
import time
from uuid import uuid4

from parea.cache.cache import Cache
from parea.constants import PAREA_OS_ENV_EXPERIMENT_UUID, TURN_OFF_PAREA_LOGGING
from parea.evals.utils import _make_evaluations
from parea.helpers import date_and_time_string_to_timestamp
from parea.schemas.models import TraceLog
from parea.utils.trace_utils import call_eval_funcs_then_log, to_date_and_time_string, trace_context, trace_data
from parea.wrapper.utils import skip_decorator_if_func_in_stack

logger = logging.getLogger()


class Wrapper:
    def __init__(
        self,
        module: Any,
        func_names: list[str],
        resolver: Callable,
        gen_resolver: Callable,
        agen_resolver: Callable,
        should_use_gen_resolver: Callable,
        cache: Cache,
        convert_kwargs_to_cache_request: Callable,
        convert_cache_to_response: Callable,
        aconvert_cache_to_response: Callable,
        log: Callable,
    ) -> None:
        self.resolver = resolver
        self.gen_resolver = gen_resolver
        self.agen_resolver = agen_resolver
        self.should_use_gen_resolver = should_use_gen_resolver
        self.log = log
        self.wrap_functions(module, func_names)
        self.cache = cache
        self.convert_kwargs_to_cache_request = convert_kwargs_to_cache_request
        self.convert_cache_to_response = convert_cache_to_response
        self.aconvert_cache_to_response = aconvert_cache_to_response

    def wrap_functions(self, module: Any, func_names: list[str]):
        for func_name in func_names:
            func_name_parts = func_name.split(".")
            original = functools.reduce(getattr, func_name_parts, module)
            setattr(module if len(func_name_parts) == 1 else functools.reduce(getattr, func_name_parts[:-1], module), func_name_parts[-1], self._wrapped_func(original))

    def _wrapped_func(self, original_func: Callable) -> Callable:
        unwrapped_func = original_func
        while hasattr(unwrapped_func, "__wrapped__"):
            unwrapped_func = unwrapped_func.__wrapped__
        return self._get_decorator(unwrapped_func, original_func)

    def _get_decorator(self, unwrapped_func: Callable, original_func: Callable):
        if inspect.iscoroutinefunction(unwrapped_func):
            return self.async_decorator(original_func)
        else:
            return self.sync_decorator(original_func)

    def _init_trace(self) -> tuple[str, float]:
        start_time = time.time()
        trace_id = str(uuid4())
        if TURN_OFF_PAREA_LOGGING:
            return trace_id, start_time
        try:
            trace_context.get().append(trace_id)

            trace_data.get()[trace_id] = TraceLog(
                trace_id=trace_id,
                parent_trace_id=trace_id,
                root_trace_id=trace_id,
                start_timestamp=to_date_and_time_string(start_time),
                trace_name="LLM",
                end_user_identifier=None,
                metadata=None,
                target=None,
                tags=None,
                inputs={},
                experiment_uuid=os.getenv(PAREA_OS_ENV_EXPERIMENT_UUID, None),
            )

            parent_trace_id = trace_context.get()[-2] if len(trace_context.get()) > 1 else None
            if not parent_trace_id:
                # we don't have a parent trace id, so we need to create one
                parent_trace_id = str(uuid4())
                trace_context.get().insert(0, parent_trace_id)
                trace_data.get()[parent_trace_id] = TraceLog(
                    trace_id=parent_trace_id,
                    parent_trace_id=parent_trace_id,
                    root_trace_id=parent_trace_id,
                    start_timestamp=to_date_and_time_string(start_time),
                    end_user_identifier=None,
                    metadata=None,
                    target=None,
                    tags=None,
                    inputs={},
                    experiment_uuid=os.getenv(PAREA_OS_ENV_EXPERIMENT_UUID, None),
                )
            trace_data.get()[trace_id].root_trace_id = trace_context.get()[0]
            trace_data.get()[trace_id].parent_trace_id = parent_trace_id
            trace_data.get()[parent_trace_id].children.append(trace_id)
            self.log(parent_trace_id)
        except Exception as e:
            logger.debug(f"Error occurred initializing openai trace, {e}")

        return trace_id, start_time

    @skip_decorator_if_func_in_stack(call_eval_funcs_then_log, _make_evaluations)
    def async_decorator(self, orig_func: Callable) -> Callable:
        async def wrapper(*args, **kwargs):
            trace_id, start_time = self._init_trace()
            response = None
            exception = None
            error = None
            cache_hit = False
            cache_key = self.convert_kwargs_to_cache_request(args, kwargs)
            try:
                if self.cache:
                    cache_result = await self.cache.aget(cache_key)
                    if cache_result is not None:
                        response = self.aconvert_cache_to_response(args, kwargs, cache_result)
                        cache_hit = True
                if response is None:
                    response = await orig_func(*args, **kwargs)
            except Exception as e:
                exception = e
                error = str(e)
                if self.cache:
                    await self.cache.ainvalidate(cache_key)
            finally:
                if exception is not None:
                    self._acleanup_trace(trace_id, start_time, error, cache_hit, args, kwargs, response)
                    raise exception
                else:
                    return self._acleanup_trace(trace_id, start_time, error, cache_hit, args, kwargs, response)

        return wrapper

    @skip_decorator_if_func_in_stack(call_eval_funcs_then_log, _make_evaluations)
    def sync_decorator(self, orig_func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            trace_id, start_time = self._init_trace()
            response = None
            error = None
            cache_hit = False
            cache_key = self.convert_kwargs_to_cache_request(args, kwargs)
            exception = None
            try:
                if self.cache:
                    cache_result = self.cache.get(cache_key)
                    if cache_result is not None:
                        response = self.convert_cache_to_response(args, kwargs, cache_result)
                        cache_hit = True
                if response is None:
                    response = orig_func(*args, **kwargs)
            except Exception as e:
                exception = e
                error = str(e)
                if self.cache:
                    self.cache.invalidate(cache_key)
            finally:
                if exception is not None:
                    self._cleanup_trace(trace_id, start_time, error, cache_hit, args, kwargs, response)
                    raise exception
                else:
                    return self._cleanup_trace(trace_id, start_time, error, cache_hit, args, kwargs, response)

        return wrapper

    def _cleanup_trace_core(self, trace_id: str, start_time: float, error: str, cache_hit, args, kwargs):
        trace_data.get()[trace_id].cache_hit = cache_hit

        if error:
            trace_data.get()[trace_id].error = error
            trace_data.get()[trace_id].status = "error"
        else:
            trace_data.get()[trace_id].status = "success"

        def final_log():
            end_time = time.time()
            trace_data.get()[trace_id].end_timestamp = to_date_and_time_string(end_time)
            trace_data.get()[trace_id].latency = end_time - start_time

            parent_id = trace_context.get()[-2]
            trace_data.get()[parent_id].end_timestamp = to_date_and_time_string(end_time)
            start_time_parent = date_and_time_string_to_timestamp(trace_data.get()[parent_id].start_timestamp)
            trace_data.get()[parent_id].latency = end_time - start_time_parent

            if not error and self.cache:
                self.cache.set(self.convert_kwargs_to_cache_request(args, kwargs), trace_data.get()[trace_id])

            self.log(trace_id)
            self.log(parent_id)
            trace_context.get().pop()

        return final_log

    def _cleanup_trace(self, trace_id: str, start_time: float, error: str, cache_hit, args, kwargs, response):
        try:
            final_log = self._cleanup_trace_core(trace_id, start_time, error, cache_hit, args, kwargs)

            if self.should_use_gen_resolver(response):
                return self.gen_resolver(trace_id, args, kwargs, response, final_log)
            else:
                self.resolver(trace_id, args, kwargs, response)
                final_log()
                return response
        except Exception as e:
            logger.debug(f"Error occurred cleaning up openai trace, {e}")
            return response

    def _acleanup_trace(self, trace_id: str, start_time: float, error: str, cache_hit, args, kwargs, response):
        try:
            final_log = self._cleanup_trace_core(trace_id, start_time, error, cache_hit, args, kwargs)

            if self.should_use_gen_resolver(response):
                return self.agen_resolver(trace_id, args, kwargs, response, final_log)
            else:
                self.resolver(trace_id, args, kwargs, response)
                final_log()
                return response
        except Exception as e:
            logger.debug(f"Error occurred cleaning up openai trace, {e}")
            return response
