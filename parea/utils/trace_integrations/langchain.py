from typing import Any, Dict, List, Optional, Union

import logging
from uuid import UUID

from langchain_core.tracers import BaseTracer
from langchain_core.tracers.schemas import ChainRun, LLMRun, Run, ToolRun

from parea.helpers import is_logging_disabled
from parea.parea_logger import parea_logger
from parea.schemas import UpdateTraceScenario
from parea.schemas.log import TraceIntegrations
from parea.utils.trace_utils import fill_trace_data, get_current_trace_id, get_root_trace_id

logger = logging.getLogger()


class PareaAILangchainTracer(BaseTracer):
    parent_trace_id: UUID
    _parea_root_trace_id: str = None
    _parea_parent_trace_id: str = None
    _session_id: Optional[str] = None
    _tags: List[str] = []
    _metadata: Dict[str, Any] = {}
    _end_user_identifier: Optional[str] = None
    _deployment_id: Optional[str] = None

    def __init__(
        self,
        session_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        end_user_identifier: Optional[str] = None,
        deployment_id: Optional[str] = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self._session_id = session_id
        self._end_user_identifier = end_user_identifier
        self._deployment_id = deployment_id
        if tags:
            self._tags = tags
        if metadata:
            self._metadata = metadata

    def _persist_run(self, run: Union[Run, LLMRun, ChainRun, ToolRun]) -> None:
        if is_logging_disabled():
            return
        self.parent_trace_id = run.id
        # using .dict() since langchain Run class currently set to Pydantic v1
        data = run.dict()
        data["_parea_root_trace_id"] = self._parea_root_trace_id or None
        data["_session_id"] = self._session_id
        data["_tags"] = self._tags
        data["_metadata"] = self._metadata
        data["_end_user_identifier"] = self._end_user_identifier
        data["_deployment_id"] = self._deployment_id
        # check if run has an attribute execution order
        if (hasattr(run, "execution_order") and run.execution_order == 1) or run.parent_run_id is None:
            data["_parea_parent_trace_id"] = self._parea_parent_trace_id or None
        parea_logger.record_vendor_log(data, TraceIntegrations.LANGCHAIN)

    def get_parent_trace_id(self) -> UUID:
        return self.parent_trace_id

    def _on_run_create(self, run: Run) -> None:
        if (hasattr(run, "execution_order") and run.execution_order == 1) or run.parent_run_id is None:
            # need to check if any traces already exist
            self._parea_root_trace_id = get_root_trace_id()
            if parent_trace_id := get_current_trace_id():
                self._parea_parent_trace_id = parent_trace_id
                fill_trace_data(str(run.id), {"parent_trace_id": parent_trace_id}, UpdateTraceScenario.LANGCHAIN_CHILD)

    def _on_llm_end(self, run: Run) -> None:
        self._persist_run(run)

    def _on_chain_end(self, run: Run) -> None:
        self._persist_run(run)
