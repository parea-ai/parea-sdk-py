from typing import Any, Callable, List, Mapping, Tuple

import contextvars
import logging
from json import JSONDecodeError

from instructor.retry import InstructorRetryException
from pydantic import ValidationError
from wrapt import wrap_object

from parea import trace
from parea.helpers import gen_trace_id
from parea.schemas import EvaluationResult, UpdateLog
from parea.utils.trace_integrations.wrapt_utils import CopyableFunctionWrapper
from parea.utils.trace_utils import logger_update_record, trace_data, trace_insert
from parea.utils.universal_encoder import json_dumps

logger = logging.getLogger()

instructor_trace_id = contextvars.ContextVar("instructor_trace_id", default="")
instructor_val_err_count = contextvars.ContextVar("instructor_val_err_count", default=0)
instructor_val_errs = contextvars.ContextVar("instructor_val_errs", default=[])


def instrument_instructor_validation_errors() -> None:
    for retry_method in ["retry_async", "retry_sync"]:
        wrap_object(
            module="instructor.patch",
            name=f"{retry_method}",
            factory=CopyableFunctionWrapper,
            args=(_RetryWrapper(),),
        )

    wrap_object(
        module="tenacity",
        name="AttemptManager.__exit__",
        factory=CopyableFunctionWrapper,
        args=(_AttemptManagerExitWrapper(),),
    )


def get_reasons(exception: Exception) -> list[str]:
    if isinstance(exception, InstructorRetryException):
        return [str(arg) for arg in exception.args]
    return [str(exception)]


def report_instructor_validation_errors() -> None:
    reason = "\n\n\n".join(instructor_val_errs.get())
    if reason:
        reason = "\n" + reason
    instructor_score = EvaluationResult(
        name="instructor_validation_error_count",
        score=instructor_val_err_count.get(),
        reason=reason,
    )
    trace_update_dict = {"scores": [instructor_score]}
    i_trace_id = instructor_trace_id.get()
    if i_trace_id:
        if children := trace_data.get()[i_trace_id].children:
            last_child_trace_id = children[-1]
            trace_update_dict["configuration"] = trace_data.get()[last_child_trace_id].configuration
        trace_insert(trace_update_dict, i_trace_id)
    instructor_trace_id.set("")
    instructor_val_err_count.set(0)
    instructor_val_errs.set([])


class _RetryWrapper:
    def __call__(
        self,
        wrapped: Callable[..., Any],
        instance: Any,
        args: Tuple[type, Any],
        kwargs: Mapping[str, Any],
    ) -> Any:
        trace_id = gen_trace_id()
        instructor_trace_id.set(trace_id)
        try:
            inputs = kwargs.get("kwargs", {}).get("template_inputs", {})
            metadata = {}
            for key in ["max_retries", "response_model", "validation_context", "mode", "args"]:
                if kwargs.get(key):
                    metadata[key] = kwargs[key]
            trace_name = "instructor"
            if "response_model" in kwargs and kwargs["response_model"] and hasattr(kwargs["response_model"], "__name__"):
                trace_name = kwargs["response_model"].__name__

            def fn_transform_generator_outputs(items: List) -> str:
                try:
                    return json_dumps(items[-1])
                except Exception as e:
                    logger.warning(f"Failed to serialize generator output: {e}", exc_info=e)
                    return ""

            return trace(name=trace_name, overwrite_trace_id=trace_id, overwrite_inputs=inputs, metadata=metadata, fn_transform_generator_outputs=fn_transform_generator_outputs)(
                wrapped
            )(*args, **kwargs)
        except (InstructorRetryException, ValidationError, JSONDecodeError) as e:
            instructor_val_err_count.set(instructor_val_err_count.get() + 1)
            reasons = get_reasons(e)
            instructor_val_errs.set(instructor_val_errs.get() + reasons)

            try:
                report_instructor_validation_errors()
            except Exception as e:
                logger.error(f"Failed to report instructor validation errors: {e}", exc_info=e)

            current_log = trace_data.get()[trace_id]
            logger_update_record(UpdateLog(trace_id=trace_id, field_name_to_value_map={"scores": current_log.scores}, root_trace_id=current_log.root_trace_id))

            raise e


class _AttemptManagerExitWrapper:
    def __call__(
        self,
        wrapped: Callable[..., Any],
        instance: Any,
        args: Tuple[type, Any],
        kwargs: Mapping[str, Any],
    ) -> Any:
        if instructor_trace_id.get() is not None:
            if len(args) > 1 and args[1] is not None and isinstance(args[1], (InstructorRetryException, ValidationError, JSONDecodeError)):
                instructor_val_err_count.set(instructor_val_err_count.get() + 1)
                reasons = get_reasons(args[1])
                instructor_val_errs.set(instructor_val_errs.get() + reasons)
            else:
                try:
                    report_instructor_validation_errors()
                except Exception as e:
                    logger.error(f"Failed to report instructor validation errors: {e}", exc_info=e)
        return wrapped(*args, **kwargs)
