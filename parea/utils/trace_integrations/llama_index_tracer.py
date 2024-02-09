from typing import Any, Dict, List, Optional

from llama_index.callbacks import CBEventType, LlamaDebugHandler


class PareaAILlamaIndexTracer(LlamaDebugHandler):
    parent_trace_id: str

    def __init__(self) -> None:
        super().__init__(event_starts_to_ignore=[], event_ends_to_ignore=[])

    def on_event_start(
        self,
        event_type: CBEventType,
        payload: Optional[Dict[str, Any]] = None,
        event_id: str = "",
        parent_id: str = "",
        **kwargs: Any,
    ) -> str:
        return super().on_event_start(event_type, payload, event_id, parent_id, **kwargs)

    def on_event_end(
        self,
        event_type: CBEventType,
        payload: Optional[Dict[str, Any]] = None,
        event_id: str = "",
        **kwargs: Any,
    ) -> None:
        super().on_event_end(event_type, payload, event_id, **kwargs)

    def start_trace(self, trace_id: Optional[str] = None) -> None:
        super().start_trace(trace_id)

    def end_trace(
        self,
        trace_id: Optional[str] = None,
        trace_map: Optional[Dict[str, List[str]]] = None,
    ) -> None:
        self.parent_trace_id = trace_id
        print(trace_map[trace_id])

    def get_parent_trace_id(self) -> str:
        return self.parent_trace_id
