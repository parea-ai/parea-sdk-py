from typing import Any, Callable, Optional, Union

import json
import os
from collections import defaultdict
from collections.abc import AsyncIterator, Iterator, Sequence

import openai
from openai import __version__ as openai_version

if openai_version.startswith("0."):
    from openai.openai_object import OpenAIObject
    from openai.util import convert_to_openai_object
else:
    from openai.types.chat import ChatCompletion as OpenAIObject

    def convert_to_openai_object(**kwargs):
        return OpenAIObject(**kwargs)


from dotenv import load_dotenv

from ..cache.cache import Cache
from ..schemas.log import LLMInputs, ModelParams
from ..schemas.models import CacheRequest, TraceLog
from ..utils.trace_utils import trace_data
from .wrapper import Wrapper

load_dotenv()

OPENAI_MODEL_INFO: dict[str, dict[str, Union[float, int, dict[str, int]]]] = {
    "gpt-3.5-turbo": {
        "prompt": 1.5,
        "completion": 2.0,
        "token_limit": {"max_completion_tokens": 4096, "max_prompt_tokens": 4096},
    },
    "gpt-3.5-turbo-0301": {
        "prompt": 1.5,
        "completion": 4.0,
        "token_limit": {"max_completion_tokens": 4096, "max_prompt_tokens": 4096},
    },
    "gpt-3.5-turbo-0613": {
        "prompt": 1.5,
        "completion": 4.0,
        "token_limit": {"max_completion_tokens": 4096, "max_prompt_tokens": 4096},
    },
    "gpt-3.5-turbo-16k": {
        "prompt": 3.0,
        "completion": 4.0,
        "token_limit": {"max_completion_tokens": 16385, "max_prompt_tokens": 16385},
    },
    "gpt-3.5-turbo-16k-0301": {
        "prompt": 3.0,
        "completion": 4.0,
        "token_limit": {"max_completion_tokens": 16385, "max_prompt_tokens": 16385},
    },
    "gpt-3.5-turbo-16k-0613": {
        "prompt": 3.0,
        "completion": 4.0,
        "token_limit": {"max_completion_tokens": 16385, "max_prompt_tokens": 16385},
    },
    "gpt-3.5-turbo-1106": {
        "prompt": 1.0,
        "completion": 2.0,
        "token_limit": {"max_completion_tokens": 4096, "max_prompt_tokens": 4096},
    },
    "gpt-3.5-turbo-instruct": {
        "prompt": 1.5,
        "completion": 4.0,
        "token_limit": {"max_completion_tokens": 4096, "max_prompt_tokens": 4096},
    },
    "gpt-4": {
        "prompt": 30.0,
        "completion": 60.0,
        "token_limit": {"max_completion_tokens": 8192, "max_prompt_tokens": 8192},
    },
    "gpt-4-0314": {
        "prompt": 30.0,
        "completion": 60.0,
        "token_limit": {"max_completion_tokens": 8192, "max_prompt_tokens": 8192},
    },
    "gpt-4-0613": {
        "prompt": 30.0,
        "completion": 60.0,
        "token_limit": {"max_completion_tokens": 8192, "max_prompt_tokens": 8192},
    },
    "gpt-4-32k": {
        "prompt": 60.0,
        "completion": 120.0,
        "token_limit": {"max_completion_tokens": 32768, "max_prompt_tokens": 32768},
    },
    "gpt-4-32k-0314": {
        "prompt": 60.0,
        "completion": 120.0,
        "token_limit": {"max_completion_tokens": 32768, "max_prompt_tokens": 32768},
    },
    "gpt-4-32k-0613": {
        "prompt": 60.0,
        "completion": 120.0,
        "token_limit": {"max_completion_tokens": 32768, "max_prompt_tokens": 32768},
    },
    "gpt-4-vision-preview": {
        "prompt": 30.0,
        "completion": 60.0,
        "token_limit": {"max_completion_tokens": 4096, "max_prompt_tokens": 128000},
    },
    "gpt-4-1106-preview": {
        "prompt": 10.0,
        "completion": 30.0,
        "token_limit": {"max_completion_tokens": 4096, "max_prompt_tokens": 128000},
    },
}


class OpenAIWrapper:
    openai.api_key = os.getenv("OPENAI_API_KEY")

    def get_original_methods(self, module_client=openai):
        if openai_version.startswith("0."):
            original_methods = {"ChatCompletion.create": module_client.ChatCompletion.create, "ChatCompletion.acreate": module_client.ChatCompletion.acreate}
        else:
            try:
                original_methods = {"chat.completions.create": module_client.chat.completions.create}
            except openai.OpenAIError:
                original_methods = {}
        return list(original_methods.keys())

    def init(self, log: Callable, cache: Cache = None, module_client=openai):
        Wrapper(
            resolver=self.resolver,
            gen_resolver=self.gen_resolver,
            agen_resolver=self.agen_resolver,
            log=log,
            module=module_client,
            func_names=self.get_original_methods(module_client),
            cache=cache,
            convert_kwargs_to_cache_request=self.convert_kwargs_to_cache_request,
            convert_cache_to_response=self.convert_cache_to_response,
            aconvert_cache_to_response=self.aconvert_cache_to_response,
        )

    @staticmethod
    def resolver(trace_id: str, _args: Sequence[Any], kwargs: dict[str, Any], response: Optional[Any]) -> Optional[Any]:
        if response:
            output = OpenAIWrapper._get_output(response)
            if openai_version.startswith("0."):
                usage = response["usage"]
                input_tokens = usage.get("prompt_tokens", 0)
                output_tokens = usage.get("completion_tokens", 0)
                total_tokens = usage.get("total_tokens", 0)
            else:
                usage = response.usage
                input_tokens = usage.prompt_tokens
                output_tokens = usage.completion_tokens
                total_tokens = usage.total_tokens
        else:
            output = None
            input_tokens = 0
            output_tokens = 0
            total_tokens = 0

        llm_configuration = OpenAIWrapper._kwargs_to_llm_configuration(kwargs)
        model = llm_configuration.model

        model_rate = OpenAIWrapper.get_model_cost(model)
        model_completion_rate = OpenAIWrapper.get_model_cost(model, is_completion=True)
        prompt_cost = (model_rate / 1000) * (input_tokens / 1000)
        completion_cost = (model_completion_rate / 1000) * (output_tokens / 1000)
        total_cost = sum([prompt_cost, completion_cost])

        trace_data.get()[trace_id].configuration = llm_configuration
        trace_data.get()[trace_id].input_tokens = input_tokens
        trace_data.get()[trace_id].output_tokens = output_tokens
        trace_data.get()[trace_id].total_tokens = total_tokens
        trace_data.get()[trace_id].cost = total_cost
        trace_data.get()[trace_id].output = output
        return response

    @staticmethod
    def gen_resolver(trace_id: str, _args: Sequence[Any], kwargs: dict[str, Any], response: Iterator[Any], final_log) -> Iterator[Any]:
        llm_configuration = OpenAIWrapper._kwargs_to_llm_configuration(kwargs)
        trace_data.get()[trace_id].configuration = llm_configuration

        message = defaultdict(str)
        for chunk in response:
            update_dict = chunk.choices[0].delta._previous
            for key, val in update_dict.items():
                if val is not None:
                    message[key] += str(val)
            yield chunk

        trace_data.get()[trace_id].output = OpenAIWrapper._get_output(message)

        final_log()

    @staticmethod
    async def agen_resolver(trace_id: str, _args: Sequence[Any], kwargs: dict[str, Any], response: AsyncIterator[Any], final_log) -> AsyncIterator[Any]:
        llm_configuration = OpenAIWrapper._kwargs_to_llm_configuration(kwargs)
        trace_data.get()[trace_id].configuration = llm_configuration

        message = defaultdict(str)
        async for chunk in response:
            update_dict = chunk.choices[0].delta._previous
            for key, val in update_dict.items():
                if val is not None:
                    message[key] += str(val)
            yield chunk

        trace_data.get()[trace_id].output = OpenAIWrapper._get_output(message)

        final_log()

    @staticmethod
    def _kwargs_to_llm_configuration(kwargs):
        return LLMInputs(
            model=kwargs.get("model", None),
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

    @staticmethod
    def _get_output(result: Any) -> str:
        if not isinstance(result, OpenAIObject) and isinstance(result, dict):
            result = convert_to_openai_object(
                {
                    "choices": [
                        {
                            "index": 0,
                            "message": result,
                        }
                    ]
                }
            )
        response_message = result.choices[0].message
        if response_message.get("function_call", None) if openai_version.startswith("0.") else response_message.function_call:
            completion = OpenAIWrapper._format_function_call(response_message)
        else:
            completion = response_message.content.strip()
        return completion

    @staticmethod
    def _format_function_call(response_message) -> str:
        def clean_json_string(s):
            """If OpenAI responds with improper newlines and multiple quotes, this will clean it up"""
            return json.dumps(s.replace("'", '"').replace("\\n", "\\\\n"))

        if openai_version.startswith("0."):
            function_name = response_message["function_call"]["name"]
            if isinstance(response_message["function_call"]["arguments"], OpenAIObject):
                function_args = dict(response_message["function_call"]["arguments"])
            else:
                function_args = json.loads(response_message["function_call"]["arguments"])
            return json.dumps({"name": function_name, "arguments": function_args}, indent=4)

        func_obj = response_message.function_call or response_message.tool_calls
        calls = []
        if not isinstance(func_obj, list):
            func_obj = [func_obj]

        for call in func_obj:
            if call:
                body = getattr(call, "function", None) or call
                function_name = body.name
                try:
                    function_args = json.loads(body.arguments)
                except json.decoder.JSONDecodeError:
                    function_args = json.loads(clean_json_string(body.arguments))
                calls.append(json.dumps({"name": function_name, "arguments": function_args}, indent=4))
        return "\n".join(calls)

    @staticmethod
    def get_model_cost(model_name: str, is_completion: bool = False) -> float:
        model_name = model_name.lower()
        cost = OPENAI_MODEL_INFO.get(model_name, {}).get("completion" if is_completion else "prompt", None)
        if cost is None:
            msg = f"Unknown model: {model_name}. " f"Please provide a valid OpenAI model name. " f"Known models are: {', '.join(OPENAI_MODEL_INFO.keys())}"
            raise ValueError(msg)

        return cost

    @staticmethod
    def convert_kwargs_to_cache_request(_args: Sequence[Any], kwargs: dict[str, Any]) -> CacheRequest:
        return CacheRequest(
            configuration=OpenAIWrapper._kwargs_to_llm_configuration(kwargs),
        )

    @staticmethod
    def _convert_cache_to_response(_args: Sequence[Any], kwargs: dict[str, Any], cache_response: TraceLog) -> OpenAIObject:
        content = cache_response.output
        message = {"role": "assistant"}
        try:
            function_call = json.loads(content)
            if isinstance(function_call, dict) and "name" in function_call and "arguments" in function_call and len(function_call) == 2:
                message["function_call"] = function_call
                message["content"] = None
            else:
                message["content"] = content
        except json.JSONDecodeError:
            message["content"] = content

        message_field = "delta" if kwargs.get("stream", False) else "message"

        return convert_to_openai_object(
            {
                "object": "chat.completion",
                "model": cache_response.configuration["model"],
                "choices": [
                    {
                        "index": 0,
                        message_field: message,
                    }
                ],
                "usage": {
                    "prompt_tokens": cache_response.input_tokens,
                    "completion_tokens": cache_response.output_tokens,
                    "total_tokens": cache_response.total_tokens,
                },
            }
        )

    @staticmethod
    def convert_cache_to_response(_args: Sequence[Any], kwargs: dict[str, Any], cache_response: TraceLog) -> Union[OpenAIObject, Iterator[OpenAIObject]]:
        response = OpenAIWrapper._convert_cache_to_response(_args, kwargs, cache_response)
        if kwargs.get("stream", False):
            return iter([response])
        else:
            return response

    @staticmethod
    def aconvert_cache_to_response(_args: Sequence[Any], kwargs: dict[str, Any], cache_response: TraceLog) -> Union[OpenAIObject, AsyncIterator[OpenAIObject]]:
        response = OpenAIWrapper._convert_cache_to_response(_args, kwargs, cache_response)
        if kwargs.get("stream", False):

            def aiterator(iterable):
                async def gen():
                    for item in iterable:
                        yield item

                return gen()

            return aiterator([response])
        else:
            return response
