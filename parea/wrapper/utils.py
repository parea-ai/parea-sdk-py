from typing import Callable

import sys
from functools import wraps


# https://gist.github.com/JettJones/c236494013f22723c1822126df944b12
def skip_decorator_if_func_in_stack(*funcs_to_check: Callable) -> Callable:
    def decorator_wrapper(decorator: Callable) -> Callable:
        def new_decorator(self, func: Callable) -> Callable:  # Include self
            @wraps(func)
            def wrapper(*args, **kwargs):
                frame = sys._getframe().f_back
                caller_names = ""
                while frame:
                    caller_names += frame.f_code.co_name + "|"
                    frame = frame.f_back
                if any(func_to_check.__name__ in caller_names for func_to_check in funcs_to_check):
                    return func(*args, **kwargs)
                return decorator(self, func)(*args, **kwargs)  # Include self

            return wrapper

        return new_decorator

    return decorator_wrapper
