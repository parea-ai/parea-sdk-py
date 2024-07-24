from typing import Any, Dict, List, Optional, Union

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
