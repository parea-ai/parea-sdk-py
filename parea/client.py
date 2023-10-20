from typing import Callable

import asyncio
import os
import time

from attrs import asdict, define, field

from parea.api_client import HTTPClient
from parea.cache.cache import Cache
from parea.cache.redis import RedisCache
from parea.helpers import gen_trace_id
from parea.parea_logger import parea_logger
from parea.schemas.models import Completion, CompletionResponse, FeedbackRequest, UseDeployedPrompt, UseDeployedPromptResponse
from parea.utils.trace_utils import get_current_trace_id, logger_all_possible, logger_record_log, trace_data
from parea.wrapper import OpenAIWrapper

COMPLETION_ENDPOINT = "/completion"
DEPLOYED_PROMPT_ENDPOINT = "/deployed-prompt"
RECORD_FEEDBACK_ENDPOINT = "/feedback"


@define
class Parea:
    api_key: str = field(init=True, default="")
    _client: HTTPClient = field(init=False, default=HTTPClient())
    cache: Cache = field(init=True, default=None)

    def __attrs_post_init__(self):
        self._client.set_api_key(self.api_key)
        if self.api_key:
            parea_logger.set_client(self._client)
        if isinstance(self.cache, RedisCache):
            parea_logger.set_redis_cache(self.cache)
        _init_parea_wrapper(logger_all_possible, self.cache)

    def completion(self, data: Completion) -> CompletionResponse:
        inference_id = gen_trace_id()
        data.inference_id = inference_id
        r = self._client.request(
            "POST",
            COMPLETION_ENDPOINT,
            data=asdict(data),
        )
        if parent_trace_id := get_current_trace_id():
            trace_data.get()[parent_trace_id].children.append(inference_id)
            logger_record_log(parent_trace_id)
        return CompletionResponse(**r.json())

    async def acompletion(self, data: Completion) -> CompletionResponse:
        inference_id = gen_trace_id()
        data.inference_id = inference_id
        r = await self._client.request_async(
            "POST",
            COMPLETION_ENDPOINT,
            data=asdict(data),
        )
        if parent_trace_id := get_current_trace_id():
            trace_data.get()[parent_trace_id].children.append(inference_id)
            logger_record_log(parent_trace_id)
        return CompletionResponse(**r.json())

    def get_prompt(self, data: UseDeployedPrompt) -> UseDeployedPromptResponse:
        r = self._client.request(
            "POST",
            DEPLOYED_PROMPT_ENDPOINT,
            data=asdict(data),
        )
        return UseDeployedPromptResponse(**r.json())

    async def aget_prompt(self, data: UseDeployedPrompt) -> UseDeployedPromptResponse:
        r = await self._client.request_async(
            "POST",
            DEPLOYED_PROMPT_ENDPOINT,
            data=asdict(data),
        )
        return UseDeployedPromptResponse(**r.json())

    def record_feedback(self, data: FeedbackRequest) -> None:
        time.sleep(2)  # give logs time to update
        self._client.request(
            "POST",
            RECORD_FEEDBACK_ENDPOINT,
            data=asdict(data),
        )

    async def arecord_feedback(self, data: FeedbackRequest) -> None:
        await asyncio.sleep(2)  # give logs time to update
        await self._client.request_async(
            "POST",
            RECORD_FEEDBACK_ENDPOINT,
            data=asdict(data),
        )


_initialized_parea_wrapper = False


def init(api_key: str = os.getenv("PAREA_API_KEY"), cache: Cache = None) -> None:
    Parea(api_key=api_key, cache=cache)


def _init_parea_wrapper(log: Callable = None, cache: Cache = None):
    global _initialized_parea_wrapper
    if _initialized_parea_wrapper:
        return
    OpenAIWrapper().init(log=log, cache=cache)
    _initialized_parea_wrapper = True
