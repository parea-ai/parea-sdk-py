import asyncio
import os
import time
from uuid import uuid4

from attrs import asdict, define, field

from parea.api_client import HTTPClient
from parea.parea_logger import parea_logger
from parea.schemas.models import CacheRequest, Completion, CompletionResponse, FeedbackRequest, TraceLog, UseDeployedPrompt, UseDeployedPromptResponse
from parea.utils.trace_utils import default_logger, get_current_trace_id, trace_data
from parea.wrapper import OpenAIWrapper, Wrapper

COMPLETION_ENDPOINT = "/completion"
DEPLOYED_PROMPT_ENDPOINT = "/deployed-prompt"
RECORD_FEEDBACK_ENDPOINT = "/feedback"
GET_CACHE_ENDPOINT = "/get_cache"


@define
class Parea:
    api_key: str = field(init=True, default="")
    _client: HTTPClient = field(init=False, default=HTTPClient())
    debug_with_cache: bool = field(init=True, default=False)

    def __attrs_post_init__(self):
        self._client.set_api_key(self.api_key)
        parea_logger.set_client(self._client)
        _init_parea_wrapper(self, self.debug_with_cache)

    def completion(self, data: Completion) -> CompletionResponse:
        inference_id = str(uuid4())
        data.inference_id = inference_id
        r = self._client.request(
            "POST",
            COMPLETION_ENDPOINT,
            data=asdict(data),
        )
        if parent_trace_id := get_current_trace_id():
            trace_data.get()[parent_trace_id].children.append(inference_id)
            default_logger(parent_trace_id)
        return CompletionResponse(**r.json())

    async def acompletion(self, data: Completion) -> CompletionResponse:
        inference_id = str(uuid4())
        data.inference_id = inference_id
        r = await self._client.request_async(
            "POST",
            COMPLETION_ENDPOINT,
            data=asdict(data),
        )
        if parent_trace_id := get_current_trace_id():
            trace_data.get()[parent_trace_id].children.append(inference_id)
            default_logger(parent_trace_id)
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

    def get_cache(self, data: CacheRequest) -> TraceLog:
        r = self._client.request(
            "POST",
            GET_CACHE_ENDPOINT,
            data=asdict(data),
        )
        return TraceLog(**r.json())

    async def aget_cache(self, data: CacheRequest) -> TraceLog:
        r = await self._client.request_async(
            "POST",
            GET_CACHE_ENDPOINT,
            data=asdict(data),
        )
        return TraceLog(**r.json())


_initialized_parea_wrapper = False


def init(api_key: str = os.getenv("PAREA_API_KEY"), debug_with_cache: bool = False) -> None:
    Parea(api_key=api_key, debug_with_cache=debug_with_cache)


def _init_parea_wrapper(parea_client: Parea, debug_with_cache: bool = False):
    global _initialized_parea_wrapper
    if _initialized_parea_wrapper:
        return
    Wrapper._parea_client = parea_client
    OpenAIWrapper().init(log=default_logger, debug_with_cache=debug_with_cache)
    _initialized_parea_wrapper = True
