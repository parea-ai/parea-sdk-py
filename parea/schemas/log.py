from typing import Any, Dict, List, Optional, Union

import json
import math
from enum import Enum

from attrs import define, field


class TraceIntegrations(str, Enum):
    LANGCHAIN = "langchain"


class Role(str, Enum):
    user = "user"
    assistant = "assistant"
    system = "system"
    example_user = "example_user"
    example_assistant = "example_assistant"
    tool = "tool"


@define
class Message:
    content: str
    role: Role = Role.user

    def to_dict(self) -> Dict[str, str]:
        return {
            "content": self.content,
            "role": str(self.role),
        }


@define
class ModelParams:
    temp: Optional[float] = 1.0
    top_p: Optional[float] = 1.0
    frequency_penalty: Optional[float] = 0.0
    presence_penalty: Optional[float] = 0.0
    max_length: Optional[int] = None
    response_format: Optional[dict] = None
    safe_prompt: Optional[bool] = None


@define
class LLMInputs:
    """model choice should match an enabled model on the Parea platform."""

    model: Optional[str] = None
    provider: Optional[str] = None
    model_params: Optional[ModelParams] = ModelParams()
    messages: Optional[List[Message]] = None
    history: Optional[List[Message]] = None
    functions: Optional[List[Any]] = None
    function_call: Optional[Union[str, Dict[str, str]]] = None


@define
class Log:
    configuration: LLMInputs = LLMInputs()
    inputs: Optional[Dict[str, str]] = None
    output: Optional[str] = None
    target: Optional[str] = None
    latency: Optional[float] = 0.0
    time_to_first_token: Optional[float] = None
    input_tokens: Optional[int] = 0
    output_tokens: Optional[int] = 0
    total_tokens: Optional[int] = 0
    cost: Optional[float] = 0.0

    def convert_to_jsonl_row_for_finetuning(self) -> dict:
        """Converts the trace log to a row in the finetuning jsonl format."""
        jsonl_row = {"messages": [m.to_dict() for m in self.configuration.messages]}
        output = self.output
        try:
            tool_calls = json.loads(output)
            tools = self.configuration.functions
            # if 'arguments' is in the output, it was actually a function call
            if "arguments" in tool_calls:
                function_call = tool_calls[0] if isinstance(tool_calls, List) else tool_calls
                function_call["arguments"] = json.dumps(function_call["arguments"])
                assistant_response = {
                    "role": "assistant",
                    "function_call": function_call,
                }
                jsonl_row["functions"] = tools
            else:
                tool_calls = tool_calls if isinstance(tool_calls, List) else [tool_calls]
                tool_calls = [
                    {"id": tool["id"], "type": "function", "function": {"name": tool["function"]["name"], "arguments": json.dumps(tool["function"]["arguments"])}}
                    for tool in tool_calls
                ]
                assistant_response = {
                    "role": "assistant",
                    "tool_calls": tool_calls,
                }
                jsonl_row["tools"] = [{"type": "function", "function": tool} for tool in tools]
        except json.JSONDecodeError:
            assistant_response = {"role": "assistant", "content": output}
        jsonl_row["messages"].append(assistant_response)
        return jsonl_row


@define
class EvaluationResult:
    name: str
    score: float = field()
    reason: Optional[str] = None

    @score.validator
    def _check_score_is_finite(self, attribute, value):
        if not math.isfinite(value):
            raise ValueError(f"Score must be finite, got {value}")


@define
class EvaluatedLog(Log):
    scores: Optional[List[EvaluationResult]] = field(factory=list)

    def get_score(self, name: str) -> Optional[EvaluationResult]:
        for score in self.scores:
            if score.name == name:
                return score
        return None
