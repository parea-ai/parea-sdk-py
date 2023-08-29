from typing import Any, Callable, Dict, Optional, Sequence

import json

import openai

from ..schemas.models import LLMInputs, ModelParams
from ..utils.trace_utils import trace_data
from .wrapper import Wrapper

MODEL_COST_MAPPING: Dict[str, float] = {
    "gpt-4": 0.03,
    "gpt-4-0314": 0.03,
    "gpt-4-0613": 0.03,
    "gpt-4-completion": 0.06,
    "gpt-4-0314-completion": 0.06,
    "gpt-4-0613-completion": 0.06,
    "gpt-4-32k": 0.06,
    "gpt-4-32k-0314": 0.06,
    "gpt-4-32k-0613": 0.06,
    "gpt-4-32k-completion": 0.12,
    "gpt-4-32k-0314-completion": 0.12,
    "gpt-4-32k-0613-completion": 0.12,
    "gpt-3.5-turbo": 0.0015,
    "gpt-3.5-turbo-0301": 0.0015,
    "gpt-3.5-turbo-0613": 0.0015,
    "gpt-3.5-turbo-16k": 0.003,
    "gpt-3.5-turbo-16k-0613": 0.003,
    "gpt-3.5-turbo-completion": 0.002,
    "gpt-3.5-turbo-0301-completion": 0.002,
    "gpt-3.5-turbo-0613-completion": 0.004,
    "gpt-3.5-turbo-16k-completion": 0.004,
    "gpt-3.5-turbo-16k-0613-completion": 0.004,
    "text-ada-001": 0.0004,
    "ada": 0.0004,
    "text-babbage-001": 0.0005,
    "babbage": 0.0005,
    "text-curie-001": 0.002,
    "curie": 0.002,
    "text-davinci-003": 0.02,
    "text-davinci-002": 0.02,
    "code-davinci-002": 0.02,
}


class OpenAIWrapper:
    original_methods = {"ChatCompletion.create": openai.ChatCompletion.create, "ChatCompletion.acreate": openai.ChatCompletion.acreate}

    @staticmethod
    def resolver(trace_id: str, _args: Sequence[Any], kwargs: Dict[str, Any], response: Optional[Any]):
        if response:
            usage = response["usage"]
            output = OpenAIWrapper._get_output(response)
        else:
            output = None
            usage = {}

        model = kwargs.get("model", None)

        llm_inputs = LLMInputs(
            model=model,
            provider="openai",
            messages=kwargs.get("messages", None),
            functions=kwargs.get("functions", None),
            function_call=kwargs.get("function_call", None),
            model_params=ModelParams(
                temp=kwargs.get("temperature", 1.0),
                max_length=kwargs.get("max_tokens", None),
                top_p=kwargs.get("top_p", 1.0),
                frequency_penalty=kwargs.get("frequency_penalty", 0.0),
                presence_penalty=kwargs.get("presence_penalty", 0.0),
            ),
        )

        model_rate = OpenAIWrapper.get_model_cost(model)
        model_completion_rate = OpenAIWrapper.get_model_cost(model, is_completion=True)
        completion_cost = model_completion_rate * (usage.get("completion_tokens", 0) / 1000)
        prompt_cost = model_rate * (usage.get("prompt_tokens", 0) / 1000)
        total_cost = sum([prompt_cost, completion_cost])

        trace_data.get()[trace_id].configuration = llm_inputs
        trace_data.get()[trace_id].input_tokens = usage.get("prompt_tokens", 0)
        trace_data.get()[trace_id].output_tokens = usage.get("completion_tokens", 0)
        trace_data.get()[trace_id].total_tokens = usage.get("total_tokens", 0)
        trace_data.get()[trace_id].cost = total_cost
        trace_data.get()[trace_id].output = output

    @staticmethod
    def init(log: Callable):
        Wrapper(resolver=OpenAIWrapper.resolver, log=log, module=openai, func_names=list(OpenAIWrapper.original_methods.keys()))

    @staticmethod
    def _get_output(result) -> str:
        response_message = result.choices[0].message
        if response_message.get("function_call", None):
            completion = OpenAIWrapper._format_function_call(response_message)
        else:
            completion = response_message.content.strip()
        return completion

    @staticmethod
    def _format_function_call(response_message) -> str:
        function_name = response_message["function_call"]["name"]
        if isinstance(response_message["function_call"]["arguments"], openai.openai_object.OpenAIObject):
            function_args = dict(response_message["function_call"]["arguments"])
        else:
            function_args = json.loads(response_message["function_call"]["arguments"])
        return f'```{json.dumps({"name": function_name, "arguments": function_args}, indent=4)}```'

    @staticmethod
    def get_model_cost(model_name: str, is_completion: bool = False) -> float:
        model_name = model_name.lower()

        if model_name.startswith("gpt-4") and is_completion:
            model_name += "-completion"

        cost = MODEL_COST_MAPPING.get(model_name, None)
        if cost is None:
            msg = f"Unknown model: {model_name}. " f"Please provide a valid OpenAI model name. " f"Known models are: {', '.join(MODEL_COST_MAPPING.keys())}"
            raise ValueError(msg)

        return cost
