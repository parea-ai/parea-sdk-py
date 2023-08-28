import json
from typing import Any, Callable, Dict, Optional, Sequence

import openai

from .wrapper import Wrapper
from ..schemas.models import LLMInputs


class OpenAIWrapper:
    original_methods = {
        "ChatCompletion.create": openai.ChatCompletion.create,
        "ChatCompletion.acreate": openai.ChatCompletion.acreate
    }

    @staticmethod
    def resolver(_args: Sequence[Any], kwargs: Dict[str, Any], response: Optional[Any]) -> Dict:
        llm_config = {key: value for key, value in kwargs.items() if key in []}  # todo: parse kwargs properly according to LLMInputs & ModelParams
        if response:
            usage = response["usage"]
            output = OpenAIWrapper._get_output(response)
        else:
            output = None
            usage = {}

        return {
            "provider": "openai",
            "configuration": LLMInputs(**llm_config),
            "input_tokens": usage.get('prompt_tokens', 0),
            "output_tokens": usage.get('completion_tokens', 0),
            "total_tokens": usage.get('total_tokens', 0),
            "output": output,
        }

    @staticmethod
    def init(log: Callable):
        Wrapper(
            resolver=OpenAIWrapper.resolver,
            log=log,
            module=openai,
            func_names=list(OpenAIWrapper.original_methods.keys())
        )

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
