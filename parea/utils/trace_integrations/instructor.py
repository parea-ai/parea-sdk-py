from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

import contextvars

from parea.helpers import gen_trace_id
from wrapt import wrap_object

from parea import trace
# from parea.schemas import EvaluationResult, Log
from parea.utils.trace_integrations.wrapt_utils import CopyableFunctionWrapper
# from parea.utils.universal_encoder import json_dumps

instructor_trace_id = contextvars.ContextVar("instructor_trace_id", default='')
instructor_val_err_count = contextvars.ContextVar("instructor_val_err_count", default=0)
instructor_val_errs = contextvars.ContextVar("instructor_val_errs", default=[])


def instrument_instructor_validation_errors() -> None:
    for retry_method in ['retry_async', 'retry_sync']:
        print('trying to patch')
        wrap_object(
            module='instructor.patch',
            name=f"{retry_method}",
            factory=CopyableFunctionWrapper,
            args=(_RetryWrapper(),),
        )

    # wrap_object(
    #     module='tenacity',
    #     name="AttemptManager.__exit__",
    #     factory=CopyableFunctionWrapper,
    #     args=(_AttemptManagerExitWrapper(),),
    # )


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
        return trace(name='instructor', _trace_id=trace_id)(wrapped)(*args, **kwargs)


# class _AttemptManagerExitWrapper:
#     def __call__(
#         self,
#         wrapped: Callable[..., Any],
#         instance: Any,
#         args: Tuple[type, Any],
#         kwargs: Mapping[str, Any],
#     ) -> Any:
#         if instructor_trace_id.get() is not None:
#             if kwargs.get('exec_type') is not None and kwargs.get('exc_value') is not None:
#                 instructor_val_err_count.set(instructor_val_err_count.get() + 1)
#                 instructor_val_errs.set(instructor_val_errs.get().append(kwargs.get('exc_value')))
#             else:
#                 # todo: set instructor score here
#                 instructor_trace_id.set('')
#                 instructor_val_err_count.set(0)
#                 instructor_val_errs.set([])
#         return wrapped(*args, **kwargs)


# This is the tenacity.AttemptManager implementation:
#
# class AttemptManager:
#     """Manage attempt context."""
#
#     def __init__(self, retry_state: "RetryCallState"):
#         self.retry_state = retry_state
#
#     def __enter__(self) -> None:
#         pass
#
#     def __exit__(
#         self,
#         exc_type: t.Optional[t.Type[BaseException]],
#         exc_value: t.Optional[BaseException],
#         traceback: t.Optional["types.TracebackType"],
#     ) -> t.Optional[bool]:
#         if exc_type is not None and exc_value is not None:
#             self.retry_state.set_exception((exc_type, exc_value, traceback))
#             return True  # Swallow exception.
#         else:
#             # We don't have the result, actually.
#             self.retry_state.set_result(None)
#             return None
