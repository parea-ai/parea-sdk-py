from typing import Callable, Dict, Optional

import asyncio
import functools
import inspect
import logging
import time
from contextvars import ContextVar
from uuid import UUID, uuid4

from parea.parea_logger import parea_logger
from parea.schemas.models import KindEnum, StatusEnum

logger = logging.getLogger()

_current_span_context: ContextVar = ContextVar("_current_span_context", default=(None, None))  # (trace_id, parent_span)


class Span:
    def __init__(self, name: str, trace_id: UUID, parent_span_id: Optional[UUID] = None):
        self.name = name
        self.trace_id = trace_id
        self.span_id = uuid4()
        self.parent_span_id = parent_span_id
        self.children = []
        self.start_time = None
        self.end_time = None
        self.status = StatusEnum.success
        self.error_msg = ""
        self.kind = KindEnum.task

    def add_child(self, child: "Span"):
        self.children.append(child)
        if self.kind == KindEnum.task:  # If it has children, it's a chain
            self.kind = KindEnum.chain

    def set_error(self, error_msg: str):
        self.status = StatusEnum.error
        self.error_msg = error_msg


def trace(kind: KindEnum = KindEnum.task, user_id: str = None, metadata: dict[str, str] = None, log_inputs: bool = True, log_outputs: bool = True):
    def decorator_trace(func: Callable):
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            return await wrap_with_span(func, True, kind, *args, **kwargs)

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            return wrap_with_span(func, False, kind, *args, **kwargs)

        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

    async def wrap_with_span(func: Callable, is_async: bool, kind: KindEnum, *args, **kwargs):
        nonlocal user_id, metadata, log_inputs, log_outputs
        function_name = func.__qualname__

        # Retrieve current trace and parent span context, or create a new trace ID if there's none
        trace_id, parent_span_id = _current_span_context.get()
        if trace_id is None:
            trace_id = uuid4()
        current_span = Span(function_name, trace_id, parent_span_id)
        current_span.kind = kind  # Apply custom kind if provided
        _current_span_context.set((trace_id, current_span.span_id))

        # rest of the logic remains the same
        if log_inputs:
            sig = inspect.signature(func)
            parameters = sig.parameters
            inputs_args = {k: v for k, v in zip(parameters.keys(), args)}
            current_span.input_params = {"args": inputs_args, "kwargs": kwargs}

        try:
            if is_async:
                output = await func(*args, **kwargs)
            else:
                output = func(*args, **kwargs)
            if log_outputs:
                current_span.output_params = {"output": output}
            return output
        except Exception as e:
            current_span.set_error(str(e))
            logger.error(f"Error in {function_name}: {current_span.error_msg}")
            raise
        finally:
            current_span.end_time = time.time()
            current_span.latency = current_span.end_time - current_span.start_time
            _current_span_context.set((trace_id, parent_span_id))  # Revert to the parent span context

            log_entry = {
                "trace_id": current_span.trace_id,
                "span_id": current_span.span_id,
                "parent_span_id": current_span.parent_span_id,
                "kind": current_span.kind.value,  # Enum converted to value
                "function_name": function_name,
                "latency": current_span.latency,
                "user_id": user_id,
                "metadata": metadata,
                "input_params": current_span.input_params if log_inputs else None,
                "output_params": current_span.output_params if log_outputs else None,
                "status": current_span.status.value,  # Enum converted to value
                "error_msg": current_span.error_msg if current_span.status == StatusEnum.error else None,
            }

            # Send log entry to Kafka asynchronously
            if is_async:
                await parea_logger.async_log(log_entry)
            else:
                asyncio.run(parea_logger.async_log(log_entry))

    return decorator_trace
