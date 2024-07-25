from typing import Any, Callable, List, Tuple

import contextvars
import functools
import inspect
import logging
import os
from datetime import datetime
from uuid import uuid4

from parea.cache.cache import Cache
from parea.constants import PAREA_OS_ENV_EXPERIMENT_UUID
from parea.evals.utils import _make_evaluations
from parea.helpers import is_logging_disabled, timezone_aware_now
from parea.schemas.models import TraceLog, UpdateLog, UpdateTraceScenario
from parea.utils.trace_utils import call_eval_funcs_then_log, execution_order_counters, fill_trace_data, logger_update_record, trace_context, trace_data
from parea.wrapper.utils import safe_format_template_to_prompt, skip_decorator_if_func_in_stack_for_evals

logger = logging.getLogger()


class Wrapper:
    def __init__(
        self,
        module: Any,
        func_names: List[str],
        resolver: Callable,
        gen_resolver: Callable,
        agen_resolver: Callable,
        should_use_gen_resolver: Callable,
        cache: Cache,
        convert_kwargs_to_cache_request: Callable,
        convert_cache_to_response: Callable,
        aconvert_cache_to_response: Callable,
        log: Callable,
        name: str = "LLM",
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
        self.name = name

    def wrap_functions(self, module: Any, func_names: List[str]):
        for func_name in func_names:
            func_name_parts = func_name.split(".")
            original = functools.reduce(getattr, func_name_parts, module)
            setattr(module if len(func_name_parts) == 1 else functools.reduce(getattr, func_name_parts[:-1], module), func_name_parts[-1], self._wrapped_func(original))

    def _wrapped_func(self, original_func: Callable) -> Callable:
        unwrapped_func = original_func
        is_wrapped_attribute_name = "_is_already_wrapped_by_parea"
        while hasattr(unwrapped_func, "__wrapped__"):
            if getattr(unwrapped_func, is_wrapped_attribute_name, False):
                return original_func
            unwrapped_func = unwrapped_func.__wrapped__
        if getattr(unwrapped_func, is_wrapped_attribute_name, False):
            return original_func

        wrapped_func = self._get_decorator(unwrapped_func, original_func)
        setattr(wrapped_func, is_wrapped_attribute_name, True)

        return wrapped_func

    def _get_decorator(self, unwrapped_func: Callable, original_func: Callable):
        if inspect.iscoroutinefunction(unwrapped_func):
            return self.async_decorator(original_func)
        else:
            return self.sync_decorator(original_func)

    def _init_trace(self, kwargs) -> Tuple[str, datetime, contextvars.Token]:
        start_time = timezone_aware_now()
        trace_id = str(uuid4())

        new_trace_context = trace_context.get() + [trace_id]
        token = trace_context.set(new_trace_context)

        if is_logging_disabled():
            return trace_id, start_time, token

        if template_inputs := kwargs.pop("template_inputs", None):
            for m in kwargs.get("messages", []):
                if isinstance(m, dict) and "content" in m:
                    if isinstance(m["content"], str):
                        m["content"] = safe_format_template_to_prompt(m["content"], **template_inputs)
                    elif isinstance(m["content"], list):
                        for i, item in enumerate(m["content"]):
                            if isinstance(item, dict) and "text" in item:
                                m["content"][i]["text"] = safe_format_template_to_prompt(item["text"], **template_inputs)

        depth = len(new_trace_context) - 1
        root_trace_id = new_trace_context[0]

        # Get the execution order counter for the current root trace
        counters = execution_order_counters.get()
        if root_trace_id not in counters:
            counters[root_trace_id] = 0
        execution_order = counters[root_trace_id]
        counters[root_trace_id] += 1

        try:
            trace_data.get()[trace_id] = TraceLog(
                trace_id=trace_id,
                parent_trace_id=root_trace_id,
                root_trace_id=root_trace_id,
                start_timestamp=start_time.isoformat(),
                trace_name=self.name,
                end_user_identifier=None,
                session_id=None,
                metadata=None,
                target=None,
                tags=None,
                inputs=template_inputs,
                experiment_uuid=os.getenv(PAREA_OS_ENV_EXPERIMENT_UUID, None),
                depth=depth,
                execution_order=execution_order,
            )

            parent_trace_id = new_trace_context[-2] if len(new_trace_context) > 1 else None
            if parent_trace_id:
                fill_trace_data(trace_id, {"parent_trace_id": parent_trace_id}, UpdateTraceScenario.CHAIN)
        except Exception as e:
            logger.debug(f"Error occurred initializing openai trace, {e}")

        return trace_id, start_time, token

    @skip_decorator_if_func_in_stack_for_evals(call_eval_funcs_then_log, _make_evaluations)
    def async_decorator(self, orig_func: Callable) -> Callable:
        @functools.wraps(orig_func)
        async def wrapper(*args, **kwargs):
            trace_id, start_time, context_token = self._init_trace(kwargs)
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
                    self._acleanup_trace(trace_id, start_time, error, cache_hit, args, kwargs, response, context_token)
                    raise exception
                else:
                    return self._acleanup_trace(trace_id, start_time, error, cache_hit, args, kwargs, response, context_token)

        return wrapper

    @skip_decorator_if_func_in_stack_for_evals(call_eval_funcs_then_log, _make_evaluations)
    def sync_decorator(self, orig_func: Callable) -> Callable:
        @functools.wraps(orig_func)
        def wrapper(*args, **kwargs):
            trace_id, start_time, context_token = self._init_trace(kwargs)
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
                    self._cleanup_trace(trace_id, start_time, error, cache_hit, args, kwargs, response, context_token)
                    raise exception
                else:
                    return self._cleanup_trace(trace_id, start_time, error, cache_hit, args, kwargs, response, context_token)

        return wrapper

    def _cleanup_trace_core(self, trace_id: str, start_time: datetime, error: str, cache_hit, args, kwargs, context_token: contextvars.Token):
        trace_data.get()[trace_id].cache_hit = cache_hit

        if error:
            trace_data.get()[trace_id].error = error
            trace_data.get()[trace_id].status = "error"
        else:
            trace_data.get()[trace_id].status = "success"

        def final_log():
            end_time = timezone_aware_now()
            trace_data.get()[trace_id].end_timestamp = end_time.isoformat()
            trace_data.get()[trace_id].latency = (end_time - start_time).total_seconds()

            if not error and self.cache:
                self.cache.set(self.convert_kwargs_to_cache_request(args, kwargs), trace_data.get()[trace_id])

            self.log(trace_id)
            try:
                trace_context_before_reset = trace_context.get()
                trace_context.reset(context_token)
                trace_context_after_reset = trace_context.get()
                if len(trace_context_after_reset) > len(trace_context_before_reset):
                    # this can happen if this is a streaming call and the LLM client got modified (e.g. instructor)
                    # so we need to manually reset the trace context to the previous state
                    trace_context.set(trace_context_before_reset)
                    # if the parent trace didn't have any output, we can also update the output
                    if (parent_trace_id := trace_data.get()[trace_id].parent_trace_id) is not None and not trace_data.get()[parent_trace_id].output:
                        logger_update_record(
                            UpdateLog(
                                trace_id=parent_trace_id,
                                field_name_to_value_map={"output": trace_data.get()[trace_id].output},
                                root_trace_id=trace_data.get()[parent_trace_id].root_trace_id,
                            )
                        )
            except IndexError:
                pass

        return final_log

    def _cleanup_trace(self, trace_id: str, start_time: datetime, error: str, cache_hit, args, kwargs, response, context_token: contextvars.Token):
        try:
            final_log = self._cleanup_trace_core(trace_id, start_time, error, cache_hit, args, kwargs, context_token)

            if self.should_use_gen_resolver(response):
                return self.gen_resolver(trace_id, args, kwargs, response, final_log)
            else:
                self.resolver(trace_id, args, kwargs, response)
                final_log()
                return response
        except Exception as e:
            logger.debug(f"Error occurred cleaning up openai trace, {e}")
            return response

    def _acleanup_trace(self, trace_id: str, start_time: datetime, error: str, cache_hit, args, kwargs, response, context_token: contextvars.Token):
        try:
            final_log = self._cleanup_trace_core(trace_id, start_time, error, cache_hit, args, kwargs, context_token)

            if self.should_use_gen_resolver(response):
                return self.agen_resolver(trace_id, args, kwargs, response, final_log)
            else:
                self.resolver(trace_id, args, kwargs, response)
                final_log()
                return response
        except Exception as e:
            logger.debug(f"Error occurred cleaning up openai trace, {e}")
            return response
