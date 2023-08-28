import json
from typing import Any, Callable, Dict, Optional, Sequence

import openai

from .wrapper import Wrapper
from ..schemas.models import LLMInputs, ModelParams
from ..utils.trace_utils import trace_data


class OpenAIWrapper:
    original_methods = {
        "ChatCompletion.create": openai.ChatCompletion.create,
        "ChatCompletion.acreate": openai.ChatCompletion.acreate
    }

    @staticmethod
    def resolver(trace_id: str, _args: Sequence[Any], kwargs: Dict[str, Any], response: Optional[Any]) -> Dict:
        if response:
            usage = response["usage"]
            output = OpenAIWrapper._get_output(response)
        else:
            output = None
            usage = {}

        llm_inputs = LLMInputs(
            model=kwargs.get('model', None),
            provider='openai',
            messages=kwargs.get('messages', None),
            functions=kwargs.get('functions', None),
            function_call=kwargs.get('function_call', None),
            model_params=ModelParams(
                temp=kwargs.get('temperature', 1.0),
                max_length=kwargs.get('max_tokens', None),
                top_p=kwargs.get('top_p', 1.0),
                frequency_penalty=kwargs.get('frequency_penalty', 0.0),
                presence_penalty=kwargs.get('presence_penalty', 0.0),
            )
        )

        trace_data.get()[trace_id].configuration = llm_inputs
        trace_data.get()[trace_id].input_tokens = usage.get('prompt_tokens', 0)
        trace_data.get()[trace_id].output_tokens = usage.get('completion_tokens', 0)
        trace_data.get()[trace_id].total_tokens = usage.get('total_tokens', 0)
        trace_data.get()[trace_id].output = output

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
