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
    response_format: Optional[dict] = None


@define
class LLMInputs:
    """
    model choice should match an enabled model on the Parea platform.
    model options are:
    OPENAI_MODELS
        "gpt-3.5-turbo"
        "gpt-3.5-turbo-0301"
        "gpt-3.5-turbo-0613"
        "gpt-3.5-turbo-16k"
        "gpt-3.5-turbo-16k-0301"
        "gpt-3.5-turbo-16k-0613"
        "gpt-3.5-turbo-1106"
        "gpt-3.5-turbo-0125"
        "gpt-3.5-turbo-instruct"
        "gpt-4"
        "gpt-4-0314"
        "gpt-4-0613"
        "gpt-4-32k"
        "gpt-4-32k-0314"
        "gpt-4-32k-0613"
        "gpt-4-turbo-preview"
        "gpt-4-1106-preview"
        "gpt-4-0125-preview"
    You can use Azure models by providing your model name prefixed with azure_, e.g. azure_gpt-3.5-turbo
    ANTHROPIC_MODELS
        "claude-instant-1.1"
        "claude-instant-1"
        "claude-instant-1.2"
        "claude-instant-1-100k"
        "claude-instant-1.1-100k"
        "claude-1"
        "claude-2"
        "claude-1-100k"
        "claude-1.2"
        "claude-1.3"
        "claude-1.3-100k"
        "claude-2.1"
    AWS_ANTHROPIC_MODELS
        "anthropic.claude-instant-v1"
        "anthropic.claude-v1"
        "anthropic.claude-v2"
        "anthropic.claude-v2:1"
    ANYSCALE_MODELS
        "meta-llama/Llama-2-7b-chat-hf"
        "meta-llama/Llama-2-13b-chat-hf"
        "meta-llama/Llama-2-70b-chat-hf"
        "codellama/CodeLlama-34b-Instruct-hf"
        "mistralai/Mistral-7B-Instruct-v0.1"
        "mistralai/Mixtral-8x7B-Instruct-v0.1"
        "HuggingFaceH4/zephyr-7b-beta"
        "Open-Orca/Mistral-7B-OpenOrca"
    VERTEX_MODELS
        "gemini-pro"
        "text-bison@001"
        "text-bison@002"
        "text-bison"
        "text-bison-32k"
        "chat-bison@002"
        "chat-bison@001"
        "chat-bison"
        "chat-bison-32k
    """

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
