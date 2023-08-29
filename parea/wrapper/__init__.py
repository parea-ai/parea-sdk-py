from parea.utils.trace_utils import default_logger
from parea.wrapper.openai import OpenAIWrapper

_initialized_parea_wrapper = False


def init():
    global _initialized_parea_wrapper
    if _initialized_parea_wrapper:
        return

    OpenAIWrapper().init(default_logger)

    _initialized_parea_wrapper = True


init()
