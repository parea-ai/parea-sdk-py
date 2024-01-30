from typing import Callable, Dict

import asyncio
import os
import time
from collections.abc import Iterable

from attrs import asdict, define, field
from cattrs import structure
from dotenv import load_dotenv

from parea.api_client import HTTPClient
from parea.cache import InMemoryCache, RedisCache
from parea.cache.cache import Cache
from parea.constants import PAREA_OS_ENV_EXPERIMENT_UUID
from parea.helpers import gen_trace_id
from parea.parea_logger import parea_logger
from parea.schemas.models import (
    Completion,
    CompletionResponse,
    CreateExperimentRequest,
    CreateGetProjectResponseSchema,
    ExperimentSchema,
    ExperimentStatsSchema,
    FeedbackRequest,
    ProjectSchema,
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
PROJECT_ENDPOINT = "/project"


@define
class Parea:
    api_key: str = field(init=True, default=os.getenv("PAREA_API_KEY"))
    project_name: str = field(init=True, default="default")
    cache: Cache = field(init=True, default=None)
    _project: ProjectSchema = field(init=False, default=None)
    _client: HTTPClient = field(init=False, default=HTTPClient())

    def __attrs_post_init__(self):
        self._client.set_api_key(self.api_key)
        project_api_response: CreateGetProjectResponseSchema = self._create_or_get_project(self.project_name)
        if project_api_response.was_created:
            print(f"Created project {project_api_response.name}")
        self._project = structure(asdict(project_api_response), ProjectSchema)

        if self.api_key:
            parea_logger.set_client(self._client)
            parea_logger.set_project_uuid(self.project_uuid)
        if isinstance(self.cache, (RedisCache, InMemoryCache)):
            parea_logger.set_redis_cache(self.cache)
        _init_parea_wrapper(logger_all_possible, self.cache)

    def wrap_openai_client(self, client: "OpenAI") -> None:
        """Only necessary for instance client with OpenAI version >= 1.0.0"""
        OpenAIWrapper().init(log=logger_all_possible, cache=self.cache, module_client=client)

    def _add_project_uuid_to_data(self, data) -> dict:
        data_dict = asdict(data)
        data_dict["project_uuid"] = self._project.uuid
        return data_dict

    def completion(self, data: Completion) -> CompletionResponse:
        parent_trace_id = get_current_trace_id()
        inference_id = gen_trace_id()
        data.inference_id = inference_id
        data.parent_trace_id = parent_trace_id or inference_id

        data_dict = self._add_project_uuid_to_data(data)
        if experiment_uuid := os.getenv(PAREA_OS_ENV_EXPERIMENT_UUID, None):
            data_dict["experiment_uuid"] = experiment_uuid

        r = self._client.request(
            "POST",
            COMPLETION_ENDPOINT,
            data=data_dict,
        )
        if parent_trace_id:
            trace_data.get()[parent_trace_id].children.append(inference_id)
            trace_data.get()[parent_trace_id].experiment_uuid = experiment_uuid
            logger_record_log(parent_trace_id)
        return structure(r.json(), CompletionResponse)

    async def acompletion(self, data: Completion) -> CompletionResponse:
        parent_trace_id = get_current_trace_id()
        inference_id = gen_trace_id()
        data.inference_id = inference_id
        data.parent_trace_id = parent_trace_id or inference_id

        data_dict = self._add_project_uuid_to_data(data)
        if experiment_uuid := os.getenv(PAREA_OS_ENV_EXPERIMENT_UUID, None):
            data_dict["experiment_uuid"] = experiment_uuid

        r = await self._client.request_async(
            "POST",
            COMPLETION_ENDPOINT,
            data=data_dict,
        )
        if parent_trace_id:
            trace_data.get()[parent_trace_id].children.append(inference_id)
            trace_data.get()[parent_trace_id].experiment_uuid = experiment_uuid
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
            data=self._add_project_uuid_to_data(data),
        )
        return structure(r.json(), ExperimentSchema)

    async def acreate_experiment(self, data: CreateExperimentRequest) -> ExperimentSchema:
        r = await self._client.request_async(
            "POST",
            EXPERIMENT_ENDPOINT,
            data=self._add_project_uuid_to_data(data),
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

    async def afinish_experiment(self, experiment_uuid: str) -> ExperimentStatsSchema:
        r = await self._client.request_async(
            "POST",
            EXPERIMENT_FINISHED_ENDPOINT.format(experiment_uuid=experiment_uuid),
        )
        return structure(r.json(), ExperimentStatsSchema)

    def _create_or_get_project(self, name: str) -> CreateGetProjectResponseSchema:
        r = self._client.request(
            "POST",
            PROJECT_ENDPOINT,
            data=asdict(CreateExperimentRequest(name=name)),
        )
        return structure(r.json(), CreateGetProjectResponseSchema)

    @property
    def project_uuid(self) -> str:
        return self._project.uuid

    def experiment(self, name: str, data: Iterable[dict], func: Callable):
        from parea import Experiment

        return Experiment(name=name, data=data, func=func, p=self)


_initialized_parea_wrapper = False


def _init_parea_wrapper(log: Callable = None, cache: Cache = None):
    global _initialized_parea_wrapper
    if _initialized_parea_wrapper:
        return
    OpenAIWrapper().init(log=log, cache=cache)
    _initialized_parea_wrapper = True
