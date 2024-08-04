from __future__ import annotations

from typing import Any, Dict, List, Optional

import logging

from langchain_core.tracers import LangChainTracer
from langchain_core.tracers.schemas import Run

from parea import get_current_trace_id, get_root_trace_id
from parea.schemas import UpdateTraceScenario
from parea.utils.trace_integrations.parea_langchain_client import PareaLangchainClient
from parea.utils.trace_utils import fill_trace_data, trace_data

logger = logging.getLogger()


class PareaAILangchainTracer(LangChainTracer):
    """Base callback handler that can be used to handle callbacks from langchain."""

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
        self._set_parea_root_and_parent_trace_id(run)
        if self.is_streaming:
            self.client.stream_log(run)
        else:
            self.client.log()

    def _on_run_create(self, run: Run) -> None:
        self.client.create_run2(run)

    def _on_run_update(self, run: Run) -> None:
        self.client.update_run2(run)

    def on_llm_new_token(self, *args: Any, **kwargs: Any):
        super().on_llm_new_token(*args, **kwargs)
        self.is_streaming = True

    def _set_parea_root_and_parent_trace_id(self, run) -> None:
        if (hasattr(run, "execution_order") and run.execution_order == 1) or run.parent_run_id is None:
            # need to check if any traces already exist
            _parea_root_trace_id = get_root_trace_id()
            _parea_parent_trace_id = None
            _experiment_uuid = None
            if parent_trace_id := get_current_trace_id():
                _parea_parent_trace_id = parent_trace_id
                _experiment_uuid = trace_data.get()[parent_trace_id].experiment_uuid
                fill_trace_data(str(run.id), {"parent_trace_id": parent_trace_id}, UpdateTraceScenario.LANGCHAIN_CHILD)

            self.client.set_parea_root_and_parent_trace_id(_parea_root_trace_id, _parea_parent_trace_id, _experiment_uuid)
