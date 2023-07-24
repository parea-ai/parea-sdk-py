from typing import Any, Dict, List, Optional, Union

from enum import Enum

from attrs import define


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


@define
class ModelParams:
    model: str
    temp: float = 0.5
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    max_length: Optional[int] = None


@define
class LLMInputs:
    inputs: Optional[dict[str, Any]] = None
    model: Optional[str] = None
    provider: Optional[str] = None
    model_params: Optional[ModelParams] = None
    messages: Optional[list[Message]] = None
    functions: Optional[list[Any]] = None
    function_call: Optional[Union[str, dict[str, str]]] = None


@define
class Completion:
    llm_inputs: LLMInputs
    end_user_identifier: Optional[str] = None
    deployment_id: Optional[str] = None
    name: Optional[str] = None
    metadata: Optional[dict] = None
    cache: bool = True
    log_omit_inputs: bool = False
    log_omit_outputs: bool = False
    log_omit: bool = False


@define
class CompletionResponse:
    content: str
    latency: float
    input_tokens: int
    output_tokens: int
    total_tokens: int
    cost: float
    model: str
    provider: str
    cache_hit: bool
    status: str
    start_timestamp: str
    end_timestamp: str
    error: Optional[str] = None


@define
class UseDeployedPrompt:
    deployment_id: str
    inputs: Optional[dict[str, Any]] = None


@define
class Prompt:
    raw_messages: list[dict[str, Any]]
    messages: list[dict[str, Any]]
    inputs: Optional[dict[str, Any]] = None


@define
class UseDeployedPromptResponse:
    deployment_id: str
    name: Optional[str] = None
    functions: Optional[dict[str, Any]] = None
    function_call: Optional[str] = None
    prompt: Optional[Prompt] = None
    model: Optional[str] = None
    provider: Optional[str] = None
    model_params: Optional[dict[str, Any]] = None
