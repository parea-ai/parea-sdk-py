from typing import Union

from uuid import UUID

from langchain_core.tracers import BaseTracer
from langchain_core.tracers.schemas import ChainRun, LLMRun, Run, ToolRun

from parea.helpers import is_logging_disabled
from parea.parea_logger import parea_logger
from parea.schemas import UpdateTraceScenario
from parea.schemas.log import TraceIntegrations
from parea.utils.trace_utils import fill_trace_data, get_current_trace_id, get_root_trace_id


class PareaAILangchainTracer(BaseTracer):
    parent_trace_id: UUID
    _parea_root_trace_id: str = None
    _parea_parent_trace_id: str = None

    def _persist_run(self, run: Union[Run, LLMRun, ChainRun, ToolRun]) -> None:
        if is_logging_disabled():
            return
        self.parent_trace_id = run.id
        # using .dict() since langchain Run class currently set to Pydantic v1
        data = run.dict()
        data["_parea_root_trace_id"] = self._parea_root_trace_id or None
        if run.execution_order == 1:
            data["_parea_parent_trace_id"] = self._parea_parent_trace_id or None
        parea_logger.record_vendor_log(data, TraceIntegrations.LANGCHAIN)

    def get_parent_trace_id(self) -> UUID:
        return self.parent_trace_id

    def _on_run_create(self, run: Run) -> None:
        if run.execution_order == 1:
            # need to check if any traces already exist\
            self._parea_root_trace_id = get_root_trace_id()
            if parent_trace_id := get_current_trace_id():
                self._parea_parent_trace_id = parent_trace_id
                fill_trace_data(str(run.id), {"parent_trace_id": parent_trace_id}, UpdateTraceScenario.LANGCHAIN_CHILD)

    def _on_llm_end(self, run: Run) -> None:
        self._persist_run(run)

    def _on_chain_end(self, run: Run) -> None:
        self._persist_run(run)
