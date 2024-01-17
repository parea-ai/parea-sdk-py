from typing import Callable

import asyncio
import os
import time

from attrs import asdict, define, field
from cattrs import structure
from dotenv import load_dotenv

from parea.api_client import HTTPClient
from parea.cache import InMemoryCache, RedisCache
from parea.cache.cache import Cache
from parea.helpers import gen_trace_id
from parea.parea_logger import parea_logger
from parea.schemas.models import (
    Completion,
    CompletionResponse,
    CreateExperimentRequest,
    ExperimentSchema,
    ExperimentStatsSchema,
    FeedbackRequest,
    UseDeployedPrompt,
    UseDeployedPromptResponse,
)
from parea.utils.trace_utils import get_current_trace_id, logger_all_possible, logger_record_log, trace_data
from parea.wrapper import OpenAIWrapper

load_dotenv()

COMPLETION_ENDPOINT = "/completion"
DEPLOYED_PROMPT_ENDPOINT = "/deployed-prompt"
RECORD_FEEDBACK_ENDPOINT = "/feedback"
EXPERIMENT_ENDPOINT = "/experiment"
EXPERIMENT_STATS_ENDPOINT = "/experiment/{experiment_uuid}/stats"
EXPERIMENT_FINISHED_ENDPOINT = "/experiment/{experiment_uuid}/finished"


@define
class Parea:
    api_key: str = field(init=True, default=os.getenv("PAREA_API_KEY"))
    cache: Cache = field(init=True, default=None)
    _client: HTTPClient = field(init=False, default=HTTPClient())

    def __attrs_post_init__(self):
        self._client.set_api_key(self.api_key)
        if self.api_key:
            parea_logger.set_client(self._client)
        if isinstance(self.cache, (RedisCache, InMemoryCache)):
            parea_logger.set_redis_cache(self.cache)
        _init_parea_wrapper(logger_all_possible, self.cache)

    def wrap_openai_client(self, client: "OpenAI") -> None:
        """Only necessary for instance client with OpenAI version >= 1.0.0"""
        OpenAIWrapper().init(log=logger_all_possible, cache=self.cache, module_client=client)

    def completion(self, data: Completion) -> CompletionResponse:
        parent_trace_id = get_current_trace_id()
        inference_id = gen_trace_id()
        data.inference_id = inference_id
        data.parent_trace_id = parent_trace_id or inference_id

        r = self._client.request(
            "POST",
            COMPLETION_ENDPOINT,
            data=asdict(data),
        )
        if parent_trace_id:
            trace_data.get()[parent_trace_id].children.append(inference_id)
            logger_record_log(parent_trace_id)
        return structure(r.json(), CompletionResponse)

    async def acompletion(self, data: Completion) -> CompletionResponse:
        parent_trace_id = get_current_trace_id()
        inference_id = gen_trace_id()
        data.inference_id = inference_id
        data.parent_trace_id = parent_trace_id or inference_id

        r = await self._client.request_async(
            "POST",
            COMPLETION_ENDPOINT,
            data=asdict(data),
        )
        if parent_trace_id:
            trace_data.get()[parent_trace_id].children.append(inference_id)
            logger_record_log(parent_trace_id)
        return structure(r.json(), CompletionResponse)

    def get_prompt(self, data: UseDeployedPrompt) -> UseDeployedPromptResponse:
        r = self._client.request(
            "POST",
            DEPLOYED_PROMPT_ENDPOINT,
            data=asdict(data),
        )
        return structure(r.json(), UseDeployedPromptResponse)

    async def aget_prompt(self, data: UseDeployedPrompt) -> UseDeployedPromptResponse:
        r = await self._client.request_async(
            "POST",
            DEPLOYED_PROMPT_ENDPOINT,
            data=asdict(data),
        )
        return structure(r.json(), UseDeployedPromptResponse)

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

    def create_experiment(self, data: CreateExperimentRequest) -> ExperimentSchema:
        r = self._client.request(
            "POST",
            EXPERIMENT_ENDPOINT,
            data=asdict(data),
        )
        return structure(r.json(), ExperimentSchema)

    async def acreate_experiment(self, data: CreateExperimentRequest) -> ExperimentSchema:
        r = await self._client.request_async(
            "POST",
            EXPERIMENT_ENDPOINT,
            data=asdict(data),
        )
        return structure(r.json(), ExperimentSchema)

    def get_experiment_stats(self, experiment_uuid: str) -> ExperimentStatsSchema:
        r = self._client.request(
            "GET",
            EXPERIMENT_STATS_ENDPOINT.format(experiment_uuid=experiment_uuid),
        )
        return structure(r.json(), ExperimentStatsSchema)

    async def aget_experiment_stats(self, experiment_uuid: str) -> ExperimentStatsSchema:
        r = await self._client.request_async(
            "GET",
            EXPERIMENT_STATS_ENDPOINT.format(experiment_uuid=experiment_uuid),
        )
        return structure(r.json(), ExperimentStatsSchema)

    def finish_experiment(self, experiment_uuid: str) -> ExperimentStatsSchema:
        r = self._client.request(
            "POST",
            EXPERIMENT_FINISHED_ENDPOINT.format(experiment_uuid=experiment_uuid),
        )
        return structure(r.json(), ExperimentStatsSchema)

    async def afinish_experiment(self, experiment_uuid: str) -> ExperimentSchema:
        r = await self._client.request_async(
            "POST",
            EXPERIMENT_FINISHED_ENDPOINT.format(experiment_uuid=experiment_uuid),
        )
        return structure(r.json(), ExperimentStatsSchema)


_initialized_parea_wrapper = False


def _init_parea_wrapper(log: Callable = None, cache: Cache = None):
    global _initialized_parea_wrapper
    if _initialized_parea_wrapper:
        return
    OpenAIWrapper().init(log=log, cache=cache)
    _initialized_parea_wrapper = True
