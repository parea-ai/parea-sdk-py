from typing import Union

import os
from uuid import UUID

from langchain_core.tracers import BaseTracer
from langchain_core.tracers.schemas import ChainRun, LLMRun, Run, ToolRun

from parea.constants import PAREA_OS_ENV_EXPERIMENT_UUID
from parea.parea_logger import parea_logger
from parea.schemas.log import TraceIntegrations


class PareaAILangchainTracer(BaseTracer):
    parent_trace_id: UUID

    def _persist_run(self, run: Union[Run, LLMRun, ChainRun, ToolRun]) -> None:
        self.parent_trace_id = run.id
        # using .dict() since langchain Run class currently set to Pydantic v1
        data = run.dict()
        if experiment_uuid := os.getenv(PAREA_OS_ENV_EXPERIMENT_UUID, None):
            data["experiment_uuid"] = experiment_uuid
        parea_logger.record_vendor_log(data, TraceIntegrations.LANGCHAIN)

    def get_parent_trace_id(self) -> UUID:
        return self.parent_trace_id
