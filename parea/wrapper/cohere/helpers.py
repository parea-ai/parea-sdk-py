from typing import Any, Dict, List, Optional, Tuple, Union

import functools

import cohere
from attrs import asdict, define
from cohere import ApiMetaBilledUnits, NonStreamedChatResponse, RerankResponse

from parea.constants import COHERE_MODEL_INFO, COHERE_SEARCH_MODELS
from parea.schemas import Message, Role
from parea.utils.universal_encoder import json_dumps

DEFAULT_MODEL = "command-r-plus"
DEFAULT_TEMPERATURE = 0.3
DEFAULT_P = 0.75


@define
class CohereOutput:
    text: Optional[str] = None
    citations: Optional[str] = None
    documents: Optional[str] = None
    search_queries: Optional[str] = None
    search_results: Optional[str] = None


def chat_history_to_messages(result: NonStreamedChatResponse, **kwargs) -> list[Message]:
    messages: list[Message] = []
    if sys_message := kwargs.get("preamble", ""):
        messages.append(Message(content=sys_message, role=Role.system))
    if history := kwargs.get("chat_history", []):
        messages.extend(to_messages(history))

    messages.extend(to_messages([m.dict() for m in result.chat_history]))
    return messages


def to_messages(chat_history: List[Union[Dict, cohere.Message]]) -> List[Message]:
    role_map = {"USER": Role.user, "CHATBOT": Role.assistant, "SYSTEM": Role.system, "TOOL": Role.tool}

    def process_message(message: Union[Dict, cohere.Message]) -> Message:
        if isinstance(message, dict):
            role = role_map.get(message["role"], Role.user)
            content = message.get("message", "")
            tool_calls = message.get("tool_calls") or message.get("tool_results")
        else:  # cohere.Message
            role = role_map.get(message.role, Role.user)
            content = "" if role == Role.tool else message.message
            tool_calls = getattr(message, "tool_calls", None) or getattr(message, "tool_results", None)

        if tool_calls:
            tc = json_dumps([t.dict() if hasattr(t, "dict") else t for t in tool_calls])
            content = tc if role == Role.tool or not content else json_dumps({"message": content, "tool_calls": tc})

        return Message(content=content, role=role)

    return list(map(process_message, chat_history))


@functools.lru_cache(maxsize=128)
def compute_cost(prompt_tokens: int, completion_tokens: int, search_units: int, is_search_model: bool, model: str) -> float:
    cost_per_token = COHERE_MODEL_INFO.get(model, {"prompt": 0, "completion": 0})
    cost = ((prompt_tokens * cost_per_token["prompt"]) + (completion_tokens * cost_per_token["completion"])) / 1_000_000
    if is_search_model:
        cost += search_units * cost_per_token.get("search", 0) / 1_000
    cost = round(cost, 10)
    return cost


def get_usage_stats(result: Optional[NonStreamedChatResponse | RerankResponse], model: str) -> Tuple[int, int, float]:
    bu: Optional[ApiMetaBilledUnits] = result.meta.billed_units if result else None
    if not bu:
        return 0, 0, 0.0
    prompt_tokens = bu.input_tokens or 0
    completion_tokens = bu.output_tokens or 0
    search_units = bu.search_units or 0
    is_search_model: bool = model in COHERE_SEARCH_MODELS
    cost = compute_cost(prompt_tokens, completion_tokens, search_units, is_search_model, model)
    return prompt_tokens, completion_tokens, cost


def get_output(result: Optional[NonStreamedChatResponse | RerankResponse]) -> str:
    if not result:
        return ""

    if isinstance(result, RerankResponse):
        output = CohereOutput(documents=cohere_json_list(result.results) if result.results else None)
        return json_dumps(asdict(output))

    text = result.text or cohere_json_list(result.tool_calls)
    output = CohereOutput(
        text=text,
        citations=cohere_json_list(result.citations) if result.citations else None,
        documents=cohere_json_list(result.documents) if result.documents else None,
        search_queries=cohere_json_list(result.search_queries) if result.search_queries else None,
        search_results=cohere_json_list(result.search_results) if result.search_results else None,
    )
    return json_dumps(asdict(output))


def cohere_json_list(obj: Any) -> str:
    out = []
    for o in obj or []:
        if isinstance(o, dict):
            out.append(o)
        else:
            out.append(o.dict())
    return json_dumps(out)
