from typing import Any, Callable, Mapping, Tuple

import contextvars

from instructor.retry import InstructorRetryException
from wrapt import wrap_object

from parea import trace
from parea.helpers import gen_trace_id
from parea.schemas import EvaluationResult, UpdateLog
from parea.utils.trace_integrations.wrapt_utils import CopyableFunctionWrapper
from parea.utils.trace_utils import logger_update_record, trace_data, trace_insert

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


def report_instructor_validation_errors() -> None:
    reason = "\n\n\n".join(instructor_val_errs.get())
    if reason:
        reason = "\n" + reason
    instructor_score = EvaluationResult(
        name="instructor_validation_error_count",
        score=instructor_val_err_count.get(),
        reason=reason,
    )
    last_child_trace_id = trace_data.get()[instructor_trace_id.get()].children[-1]
    trace_insert(
        {
            "scores": [instructor_score],
            "configuration": trace_data.get()[last_child_trace_id].configuration,
        },
        instructor_trace_id.get(),
    )
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
            return trace(
                name="instructor",
                overwrite_trace_id=trace_id,
                overwrite_inputs=inputs,
                metadata=metadata,
            )(
                wrapped
            )(*args, **kwargs)
        except InstructorRetryException as e:
            instructor_val_err_count.set(instructor_val_err_count.get() + 1)
            reasons = []
            for arg in e.args:
                reasons.append(str(arg))
            instructor_val_errs.set(instructor_val_errs.get() + reasons)

            report_instructor_validation_errors()
            logger_update_record(UpdateLog(trace_id=trace_id, field_name_to_value_map={"scores": trace_data.get()[trace_id].scores}))

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
            if len(args) > 1 and args[1] is not None and isinstance(args[1], InstructorRetryException):
                instructor_val_err_count.set(instructor_val_err_count.get() + 1)
                reasons = []
                for arg in args[1].args:
                    reasons.append(str(arg))
                instructor_val_errs.set(instructor_val_errs.get() + reasons)
            else:
                report_instructor_validation_errors()
        return wrapped(*args, **kwargs)
