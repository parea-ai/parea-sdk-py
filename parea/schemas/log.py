from typing import Any, Optional, Union

from enum import Enum

from attr import define


class TraceIntegrations(str, Enum):
    LANGCHAIN = "langchain"


class Role(str, Enum):
    user = "user"
    assistant = "assistant"
    system = "system"
    example_user = "example_user"
    example_assistant = "example_assistant"


@define
class Message:
    content: str
    role: Role = Role.user

    def to_dict(self) -> dict[str, str]:
        return {
            "content": self.content,
            "role": str(self.role),
        }


@define
class ModelParams:
    temp: float = 1.0
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    max_length: Optional[int] = None


@define
class LLMInputs:
    model: Optional[str] = None
    provider: Optional[str] = None
    model_params: Optional[ModelParams] = None
    messages: Optional[list[Message]] = None
    functions: Optional[list[Any]] = None
    function_call: Optional[Union[str, dict[str, str]]] = None


@define
class Log:
    configuration: LLMInputs = LLMInputs()
    inputs: Optional[dict[str, str]] = None
    output: Optional[str] = None
    target: Optional[str] = None
    latency: Optional[float] = 0.0
    input_tokens: Optional[int] = 0
    output_tokens: Optional[int] = 0
    total_tokens: Optional[int] = 0
    cost: Optional[float] = 0.0
