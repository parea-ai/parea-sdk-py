from typing import Callable

import inspect
from functools import wraps


def skip_decorator_if_func_in_stack(func_to_check: Callable) -> Callable:
    def decorator_wrapper(decorator: Callable) -> Callable:
        def new_decorator(self, func: Callable) -> Callable:  # Include self
            @wraps(func)
            def wrapper(*args, **kwargs):
                if any(func_to_check.__name__ in frame.function for frame in inspect.stack()):
                    return func(*args, **kwargs)
                return decorator(self, func)(*args, **kwargs)  # Include self

            return wrapper

        return new_decorator

    return decorator_wrapper
