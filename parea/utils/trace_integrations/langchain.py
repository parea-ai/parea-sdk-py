from __future__ import annotations

from typing import Any, Dict, List, Optional

import logging
from uuid import UUID

from langchain_core.tracers import LangChainTracer
from langchain_core.tracers.schemas import Run

from parea import get_current_trace_id, get_root_trace_id
from parea.helpers import is_logging_disabled
from parea.schemas import UpdateTraceScenario
from parea.utils.trace_integrations.parea_langchain_client import PareaLangchainClient
from parea.utils.trace_utils import fill_trace_data, trace_data

logger = logging.getLogger()


class PareaAILangchainTracer(LangChainTracer):
    """Base callback handler that can be used to handle callbacks from langchain."""

    parent_trace_id: UUID

    def __init__(
        self,
        session_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        end_user_identifier: Optional[str] = None,
        deployment_id: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the Parea tracer."""
        super().__init__(**kwargs)
        self.client = PareaLangchainClient(session_id, tags, metadata, end_user_identifier, deployment_id)
        self.is_streaming = False

    def _persist_run(self, run: Run) -> None:
        if is_logging_disabled():
            return
        try:
            self._set_parea_root_and_parent_trace_id(run)
            if self.is_streaming:
                self.client.stream_log(run)
            else:
                self.client.log()
        except Exception as e:
            logger.exception(f"Error persisting langchain run: {e}")

    def _on_run_create(self, run: Run) -> None:
        try:
            self.client.create_run_trace(run)
        except Exception as e:
            logger.exception(f"Error creating langchain run: {e}")

    def _on_run_update(self, run: Run) -> None:
        try:
            self.client.update_run_trace(run)
        except Exception as e:
            logger.exception(f"Error updating langchain run: {e}")

    def on_llm_new_token(self, *args: Any, **kwargs: Any):
        super().on_llm_new_token(*args, **kwargs)
        self.is_streaming = True

    def _set_parea_root_and_parent_trace_id(self, run) -> None:
        self.parent_trace_id = run.id
        if (hasattr(run, "execution_order") and run.execution_order == 1) or run.parent_run_id is None:
            parea_root_trace_id = get_root_trace_id()
            if parent_trace_id := get_current_trace_id():
                _experiment_uuid = trace_data.get()[parent_trace_id].experiment_uuid
                fill_trace_data(str(run.id), {"parent_trace_id": parent_trace_id}, UpdateTraceScenario.LANGCHAIN_CHILD)
                langchain_to_parea_root_data = {run.id: {"parent_trace_id": parent_trace_id, "root_trace_id": parea_root_trace_id, "experiment_uuid": _experiment_uuid}}
                self.client.set_parea_root_and_parent_trace_id(langchain_to_parea_root_data)

    def get_parent_trace_id(self) -> UUID:
        return self.parent_trace_id
