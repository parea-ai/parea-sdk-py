from typing import Any, Callable, Optional, Union

import json
import os
from collections import defaultdict
from collections.abc import AsyncIterator, Iterator, Sequence

import openai
from openai import __version__ as openai_version

from ..utils.universal_encoder import json_dumps
from .utils import _calculate_input_tokens, _compute_cost, _format_function_call, _kwargs_to_llm_configuration, _num_tokens_from_string

if openai_version.startswith("0."):
    from openai.openai_object import OpenAIObject
    from openai.util import convert_to_openai_object
else:
    from openai.types.chat import ChatCompletion as OpenAIObject

    def convert_to_openai_object(kwargs):
        if "id" not in kwargs:
            kwargs["id"] = "0"
        if "created" not in kwargs:
            kwargs["created"] = 0
        if "object" not in kwargs:
            kwargs["object"] = "chat.completion"
        if "model" not in kwargs:
            kwargs["model"] = "model"
        if "choices" in kwargs and isinstance(kwargs["choices"], list) and len(kwargs["choices"]) > 0:
            if "finish_reason" not in kwargs["choices"][0]:
                kwargs["choices"][0]["finish_reason"] = "stop"
        return OpenAIObject(**kwargs)


from dotenv import load_dotenv

from ..cache.cache import Cache
from ..schemas.models import CacheRequest, TraceLog
from ..utils.trace_utils import trace_data
from .wrapper import Wrapper

load_dotenv()

is_old_openai = openai_version.startswith("0.")


class OpenAIWrapper:
    openai.api_key = os.getenv("OPENAI_API_KEY")

    def get_original_methods(self, module_client=openai):
        if is_old_openai:
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
            should_use_gen_resolver=self.should_use_gen_resolver,
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
        model = response.model or llm_configuration.model

        trace_data.get()[trace_id].configuration = llm_configuration
        trace_data.get()[trace_id].input_tokens = input_tokens
        trace_data.get()[trace_id].output_tokens = output_tokens
        trace_data.get()[trace_id].total_tokens = total_tokens
        trace_data.get()[trace_id].cost = _compute_cost(input_tokens, output_tokens, model)
        trace_data.get()[trace_id].output = output
        return response

    @staticmethod
    def gen_resolver(trace_id: str, _args: Sequence[Any], kwargs: dict[str, Any], response, final_log):
        llm_configuration = OpenAIWrapper._kwargs_to_llm_configuration(kwargs)
        trace_data.get()[trace_id].configuration = llm_configuration

        accumulator = OpenAIWrapper._get_default_dict_streaming()
        for chunk in response:
            OpenAIWrapper._update_accumulator_streaming(accumulator, chunk)
            yield chunk

        trace_data.get()[trace_id].output = OpenAIWrapper._get_output(accumulator)

        model = accumulator.get("model") or llm_configuration.model
        output_tokens = _num_tokens_from_string(accumulator.get("content") or json_dumps(accumulator.get("function_call")), model)
        input_tokens = _calculate_input_tokens(
            kwargs.get("messages", []),
            kwargs.get("functions", []) or [d["function"] for d in kwargs.get("tools", [])],
            kwargs.get("function_call", "auto") or kwargs.get("tool_choice", "auto"),
            model,
        )
        trace_data.get()[trace_id].input_tokens = input_tokens
        trace_data.get()[trace_id].output_tokens = output_tokens
        trace_data.get()[trace_id].total_tokens = input_tokens + output_tokens
        trace_data.get()[trace_id].cost = _compute_cost(input_tokens, output_tokens, model)

        final_log()

    @staticmethod
    def _get_default_dict_streaming():
        return defaultdict(content="", role="", function_call=defaultdict(arguments="", name=""))

    @staticmethod
    def _update_accumulator_streaming(accumulator: defaultdict, chunk):
        if chunk.model:
            accumulator["model"] = chunk.model
        if not chunk.choices:
            return
        if is_old_openai:
            delta_dict = chunk.choices[0].delta._previous
        else:
            delta_dict = chunk.choices[0].delta.model_dump()
        if delta_dict.pop("tool_calls", None) is not None:
            raise NotImplementedError("Tool calls are not supported yet for streaming. Please, contact the team")
        for key, val in delta_dict.items():
            if val:
                if isinstance(val, dict):
                    for k, v in val.items():
                        if v:
                            accumulator[key][k] += str(v)
                elif isinstance(val, str):
                    accumulator[key] += val

    @staticmethod
    async def agen_resolver(trace_id: str, _args: Sequence[Any], kwargs: dict[str, Any], response, final_log):
        llm_configuration = OpenAIWrapper._kwargs_to_llm_configuration(kwargs)
        trace_data.get()[trace_id].configuration = llm_configuration

        accumulator = OpenAIWrapper._get_default_dict_streaming()
        async for chunk in response:
            OpenAIWrapper._update_accumulator_streaming(accumulator, chunk)
            yield chunk

        trace_data.get()[trace_id].output = OpenAIWrapper._get_output(accumulator)
        model = accumulator.get("model") or llm_configuration.model
        output_tokens = _num_tokens_from_string(accumulator.get("content") or json_dumps(accumulator.get("function_call")), model)
        input_tokens = _calculate_input_tokens(
            kwargs.get("messages", []),
            kwargs.get("functions", []) or [d["function"] for d in kwargs.get("tools", [])],
            kwargs.get("function_call", "auto") or kwargs.get("tool_choice", "auto"),
            model,
        )
        trace_data.get()[trace_id].input_tokens = input_tokens
        trace_data.get()[trace_id].output_tokens = output_tokens
        trace_data.get()[trace_id].total_tokens = input_tokens + output_tokens
        trace_data.get()[trace_id].cost = _compute_cost(input_tokens, output_tokens, model)

        final_log()

    @staticmethod
    def _kwargs_to_llm_configuration(kwargs):
        return _kwargs_to_llm_configuration(kwargs)

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
        if not response_message.get("content", None) if is_old_openai else not response_message.content:
            completion = OpenAIWrapper._format_function_call(response_message)
        else:
            completion = response_message.content.strip()
        return completion

    @staticmethod
    def _format_function_call(response_message) -> str:
        return _format_function_call(response_message)

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

    @staticmethod
    def should_use_gen_resolver(response: Any) -> bool:
        if is_old_openai:
            return isinstance(response, (Iterator, AsyncIterator))
        else:
            from openai import AsyncStream, Stream

            return isinstance(response, (Stream, AsyncStream))
