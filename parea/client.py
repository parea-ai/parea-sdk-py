from typing import Any, AsyncIterable, Callable, Dict, Iterable, List, Optional, Union

import asyncio
import logging
import os
import time

import httpx
from attrs import asdict, define, field
from cattrs import structure
from dotenv import load_dotenv

from parea.api_client import HTTPClient
from parea.cache.cache import Cache
from parea.constants import PAREA_OS_ENV_EXPERIMENT_UUID
from parea.experiment.datasets import create_test_cases, create_test_collection
from parea.helpers import gen_trace_id, serialize_metadata_values, structure_trace_log_from_api, structure_trace_logs_from_api
from parea.parea_logger import parea_logger
from parea.schemas.models import (
    Completion,
    CompletionResponse,
    CreateExperimentRequest,
    CreateGetProjectResponseSchema,
    CreateTestCaseCollection,
    CreateTestCases,
    ExperimentSchema,
    ExperimentStatsSchema,
    ExperimentWithPinnedStatsSchema,
    FeedbackRequest,
    FinishExperimentRequestSchema,
    ListExperimentUUIDsFilters,
    ProjectSchema,
    TestCaseCollection,
    TraceLog,
    TraceLogFilters,
    UseDeployedPrompt,
    UseDeployedPromptResponse,
)
from parea.utils.trace_utils import get_current_trace_id, get_root_trace_id, logger_all_possible, logger_record_log, trace_data

load_dotenv()

logger = logging.getLogger()

COMPLETION_ENDPOINT = "/completion"
DEPLOYED_PROMPT_ENDPOINT = "/deployed-prompt"
RECORD_FEEDBACK_ENDPOINT = "/feedback"
EXPERIMENT_ENDPOINT = "/experiment"
EXPERIMENT_STATS_ENDPOINT = "/experiment/{experiment_uuid}/stats"
EXPERIMENT_FINISHED_ENDPOINT = "/experiment/{experiment_uuid}/finished"
PROJECT_ENDPOINT = "/project"
GET_COLLECTION_ENDPOINT = "/collection/{test_collection_identifier}"
CREATE_COLLECTION_ENDPOINT = "/collection"
ADD_TEST_CASES_ENDPOINT = "/testcases"
GET_TRACE_LOG_ENDPOINT = "/trace_log/{trace_id}"
LIST_EXPERIMENTS_ENDPOINT = "/experiments"
GET_EXPERIMENT_LOGS_ENDPOINT = "/experiment/{experiment_uuid}/trace_logs"


@define
class Parea:
    api_key: str = field(init=True, default=os.getenv("PAREA_API_KEY"))
    project_name: str = field(init=True, default="default")
    cache: Cache = field(init=True, default=None)
    _project: ProjectSchema = field(init=False, default=None)
    _client: HTTPClient = field(init=False, default=HTTPClient())

    def __attrs_post_init__(self):
        self._client.set_api_key(self.api_key)

        if self.api_key:
            project_api_response: CreateGetProjectResponseSchema = self._create_or_get_project(self.project_name)
            if project_api_response.was_created:
                print(f"Created project {project_api_response.name}")
            self._project = structure(asdict(project_api_response), ProjectSchema)
            parea_logger.set_client(self._client)
            parea_logger.set_project_uuid(self.project_uuid)

    def wrap_openai_client(self, client: "OpenAI", integration: Optional[str] = None) -> None:
        """Only necessary for instance client with OpenAI version >= 1.0.0"""
        from parea.wrapper import OpenAIWrapper
        from parea.wrapper.openai_beta_wrapper import BetaWrappers

        OpenAIWrapper().init(log=logger_all_possible, cache=self.cache, module_client=client)
        BetaWrappers(client).init()

        if integration:
            self._client.add_integration(integration)

    def wrap_anthropic_client(self, client: "Anthropic", integration: Optional[str] = None) -> None:
        from parea.wrapper.anthropic.anthropic import AnthropicWrapper

        AnthropicWrapper().init(log=logger_all_possible, cache=self.cache, client=client)

        if integration:
            self._client.add_integration(integration)

    def auto_trace_openai_clients(self, integration: Optional[str]) -> None:
        import openai

        openai._ModuleClient = patch_openai_client_classes(openai._ModuleClient, self)
        openai.OpenAI = patch_openai_client_classes(openai.OpenAI, self)
        openai.AsyncOpenAI = patch_openai_client_classes(openai.AsyncOpenAI, self)
        openai.AzureOpenAI = patch_openai_client_classes(openai.AzureOpenAI, self)
        openai.AsyncAzureOpenAI = patch_openai_client_classes(openai.AsyncAzureOpenAI, self)

        if integration:
            self._client.add_integration(integration)

    def trace_dspy(self):
        from parea.utils.trace_integrations.dspy import DSPyInstrumentor

        try:
            import openai

            if openai.version.__version__.startswith("0."):
                self.wrap_openai_client(openai, "dspy")
            else:
                self.auto_trace_openai_clients("dspy")
        except ImportError:
            pass

        DSPyInstrumentor().instrument()

    def integrate_with_sglang(self):
        self.auto_trace_openai_clients()
        self._client.add_integration("sglang")

    def _add_project_uuid_to_data(self, data) -> dict:
        data_dict = asdict(data)
        data_dict["project_uuid"] = self._project.uuid
        return data_dict

    @property
    def project_uuid(self) -> str:
        return self._project.uuid

    def completion(self, data: Completion) -> CompletionResponse:
        data = self._update_data_and_trace(data)
        r = self._client.request(
            "POST",
            COMPLETION_ENDPOINT,
            data=asdict(data),
        )
        return structure(r.json(), CompletionResponse)

    async def acompletion(self, data: Completion) -> CompletionResponse:
        data = self._update_data_and_trace(data)
        r = await self._client.request_async(
            "POST",
            COMPLETION_ENDPOINT,
            data=asdict(data),
        )
        return structure(r.json(), CompletionResponse)

    def stream(self, data: Completion) -> Iterable[bytes]:
        data = self._update_data_and_trace(data)
        response = self._client.stream_request(
            "POST",
            f"{COMPLETION_ENDPOINT}/stream",
            data=asdict(data),
        )
        yield from response

    async def astream(self, data: Completion) -> AsyncIterable[bytes]:
        data = self._update_data_and_trace(data)
        response = self._client.stream_request_async(
            "POST",
            f"{COMPLETION_ENDPOINT}/stream",
            data=asdict(data),
        )
        async for chunk in response:
            yield chunk

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
        if not data.trace_id:
            logger.info("No trace_id found in feedback request")
            return

        time.sleep(2)  # give logs time to update
        self._client.request(
            "POST",
            RECORD_FEEDBACK_ENDPOINT,
            data=asdict(data),
        )

    async def arecord_feedback(self, data: FeedbackRequest) -> None:
        if not data.trace_id:
            logger.info("No trace_id found in feedback request")
            return

        await asyncio.sleep(2)  # give logs time to update
        await self._client.request_async(
            "POST",
            RECORD_FEEDBACK_ENDPOINT,
            data=asdict(data),
        )

    def create_experiment(self, data: CreateExperimentRequest) -> ExperimentSchema:
        try:
            r = self._client.request(
                "POST",
                EXPERIMENT_ENDPOINT,
                data=self._add_project_uuid_to_data(data),
            )
            return structure(r.json(), ExperimentSchema)
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 400:
                response_detail = e.response.json()
                error_msg = response_detail.get("detail", e.response.text)
                raise ValueError(f"Error creating experiment: {error_msg}")
            raise

    async def acreate_experiment(self, data: CreateExperimentRequest) -> ExperimentSchema:
        try:
            r = await self._client.request_async(
                "POST",
                EXPERIMENT_ENDPOINT,
                data=self._add_project_uuid_to_data(data),
            )
            return structure(r.json(), ExperimentSchema)
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 400:
                response_detail = e.response.json()
                error_msg = response_detail.get("detail", e.response.text)
                raise ValueError(f"Error creating experiment: {error_msg}")
            raise

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

    def finish_experiment(self, experiment_uuid: str, fin_eq: FinishExperimentRequestSchema) -> ExperimentStatsSchema:
        r = self._client.request(
            "POST",
            EXPERIMENT_FINISHED_ENDPOINT.format(experiment_uuid=experiment_uuid),
            data=asdict(fin_eq),
        )
        return structure(r.json(), ExperimentStatsSchema)

    async def afinish_experiment(self, experiment_uuid: str, fin_req: FinishExperimentRequestSchema) -> ExperimentStatsSchema:
        r = await self._client.request_async(
            "POST",
            EXPERIMENT_FINISHED_ENDPOINT.format(experiment_uuid=experiment_uuid),
            data=asdict(fin_req),
        )
        return structure(r.json(), ExperimentStatsSchema)

    def _create_or_get_project(self, name: str) -> CreateGetProjectResponseSchema:
        r = self._client.request(
            "POST",
            PROJECT_ENDPOINT,
            data={"name": name},
        )
        return structure(r.json(), CreateGetProjectResponseSchema)

    def get_collection(self, test_collection_identifier: Union[str, int]) -> TestCaseCollection:
        r = self._client.request(
            "GET",
            GET_COLLECTION_ENDPOINT.format(test_collection_identifier=test_collection_identifier),
        )
        return structure(r.json(), TestCaseCollection)

    async def aget_collection(self, test_collection_identifier: Union[str, int]) -> TestCaseCollection:
        r = await self._client.request_async(
            "GET",
            GET_COLLECTION_ENDPOINT.format(test_collection_identifier=test_collection_identifier),
        )
        return structure(r.json(), TestCaseCollection)

    def create_test_collection(self, data: List[Dict[str, Any]], name: Optional[str] = None) -> None:
        request: CreateTestCaseCollection = create_test_collection(data, name)
        self._client.request(
            "POST",
            CREATE_COLLECTION_ENDPOINT,
            data=asdict(request),
        )

    def add_test_cases(
        self,
        data: List[Dict[str, Any]],
        name: Optional[str] = None,
        dataset_id: Optional[int] = None,
    ) -> None:
        request = CreateTestCases(id=dataset_id, name=name, test_cases=create_test_cases(data))
        self._client.request(
            "POST",
            ADD_TEST_CASES_ENDPOINT,
            data=asdict(request),
        )

    def experiment(
        self,
        name: str,
        data: Union[str, int, Iterable[dict]],
        func: Callable,
        n_trials: int = 1,
        metadata: Optional[Dict[str, str]] = None,
        dataset_level_evals: Optional[List[Callable]] = None,
        n_workers: int = 10,
    ):
        """
        :param data: If your dataset is defined locally it should be an iterable of k/v
        pairs matching the expected inputs of your function. To reference a dataset you
        have saved on Parea, use the dataset name as a string or the dataset id as an int.
        :param name: The name of the experiment.
        :param func: The function to run. This function should accept inputs that match the keys of the data field.
        :param n_trials: The number of times to run the experiment on the same data.
        :param metadata: Optional metadata to attach to the experiment.
        :param dataset_level_evals: Optional list of functions to run on the dataset level. Each function should accept a list of EvaluatedLog objects and return a float or an EvaluationResult object
        :param n_workers: The number of workers to use for running the experiment.
        """
        from parea import Experiment

        return Experiment(
            experiment_name=name,
            data=data,
            func=func,
            p=self,
            n_trials=n_trials,
            metadata=metadata,
            dataset_level_evals=dataset_level_evals,
            n_workers=n_workers,
        )

    def _update_data_and_trace(self, data: Completion) -> Completion:
        data = serialize_metadata_values(data)
        inference_id = gen_trace_id()
        data.inference_id = inference_id
        data.project_uuid = self._project.uuid

        try:
            parent_trace_id = get_current_trace_id()
            data.parent_trace_id = parent_trace_id or inference_id
            data.root_trace_id = get_root_trace_id()

            if experiment_uuid := os.getenv(PAREA_OS_ENV_EXPERIMENT_UUID, None):
                data.experiment_uuid = experiment_uuid

            if parent_trace_id:
                trace_data.get()[parent_trace_id].children.append(inference_id)
                trace_data.get()[parent_trace_id].experiment_uuid = experiment_uuid
        except Exception as e:
            logger.debug(f"Error updating trace ids for completion. Trace log will be absent: {e}")

        return data

    def get_trace_log(self, trace_id: str) -> TraceLog:
        response = self._client.request("GET", GET_TRACE_LOG_ENDPOINT.format(trace_id=trace_id))
        return structure_trace_log_from_api(response.json())

    async def aget_trace_log(self, trace_id: str) -> TraceLog:
        response = await self._client.request_async("GET", GET_TRACE_LOG_ENDPOINT.format(trace_id=trace_id))
        return structure_trace_log_from_api(response.json())

    def list_experiments(self, filter_conditions: Optional[ListExperimentUUIDsFilters] = ListExperimentUUIDsFilters()) -> List[ExperimentWithPinnedStatsSchema]:
        response = self._client.request("POST", LIST_EXPERIMENTS_ENDPOINT, data=asdict(filter_conditions))
        return structure(response.json(), List[ExperimentWithPinnedStatsSchema])

    async def alist_experiments(self, filter_conditions: Optional[ListExperimentUUIDsFilters] = ListExperimentUUIDsFilters()) -> List[ExperimentWithPinnedStatsSchema]:
        response = await self._client.request_async("POST", LIST_EXPERIMENTS_ENDPOINT, data=asdict(filter_conditions))
        return structure(response.json(), List[ExperimentWithPinnedStatsSchema])

    def get_experiment_trace_logs(self, experiment_uuid: str, filters: TraceLogFilters = TraceLogFilters()) -> List[TraceLog]:
        response = self._client.request("POST", GET_EXPERIMENT_LOGS_ENDPOINT.format(experiment_uuid=experiment_uuid), data=asdict(filters))
        return structure_trace_logs_from_api(response.json())

    async def aget_experiment_trace_logs(self, experiment_uuid: str, filters: TraceLogFilters = TraceLogFilters()) -> List[TraceLog]:
        response = await self._client.request_async("POST", GET_EXPERIMENT_LOGS_ENDPOINT.format(experiment_uuid=experiment_uuid), data=asdict(filters))
        return structure_trace_logs_from_api(response.json())


def patch_openai_client_classes(openai_client, parea_client: Parea):
    """Creates a subclass of the given openai_client to always wrap it with Parea at instantiation."""

    def new_init(self, *args, **kwargs):
        openai_client.__init__(self, *args, **kwargs)
        parea_client.wrap_openai_client(self)

    subclass = type(openai_client.__name__, (openai_client,), {"__init__": new_init})

    return subclass
