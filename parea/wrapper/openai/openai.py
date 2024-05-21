from typing import Any, AsyncGenerator, AsyncIterator, Callable, Dict, Generator, Iterator, Optional, Sequence, TypeVar, Union

import json
import os
from collections import defaultdict
from datetime import datetime

import openai
from openai import __version__ as openai_version

from parea.helpers import timezone_aware_now
from parea.utils.universal_encoder import json_dumps
from parea.wrapper.utils import _calculate_input_tokens, _compute_cost, _format_function_call, _kwargs_to_llm_configuration, _num_tokens_from_string

if openai_version.startswith("0."):
    from openai.openai_object import OpenAIObject
    from openai.util import convert_to_openai_object
else:
    from openai.types.chat import ChatCompletion as OpenAIObject

    def convert_to_openai_object(kwargs) -> OpenAIObject:
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

from parea.cache.cache import Cache
from parea.schemas.models import CacheRequest, TraceLog
from parea.utils.trace_utils import trace_data
from parea.wrapper.wrapper import Wrapper

load_dotenv()

is_old_openai = openai_version.startswith("0.")

_T = TypeVar("_T")


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
            name="llm-openai",
        )

    def resolver(self, trace_id: str, _args: Sequence[Any], kwargs: Dict[str, Any], response: Optional[Any]) -> Optional[Any]:
        if response:
            output = self._get_output(response)
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

        llm_configuration = self._kwargs_to_llm_configuration(kwargs)
        model = response.model if response and response.model else llm_configuration.model

        trace_data.get()[trace_id].configuration = llm_configuration
        trace_data.get()[trace_id].input_tokens = input_tokens
        trace_data.get()[trace_id].output_tokens = output_tokens
        trace_data.get()[trace_id].total_tokens = total_tokens
        trace_data.get()[trace_id].cost = _compute_cost(input_tokens, output_tokens, model)
        trace_data.get()[trace_id].output = output
        return response

    def gen_resolver(self, trace_id: str, _args: Sequence[Any], kwargs: Dict[str, Any], response, final_log):
        llm_configuration = self._kwargs_to_llm_configuration(kwargs)
        trace_data.get()[trace_id].configuration = llm_configuration

        accumulator, info_from_response = self._get_default_dict_streaming()

        def gen_final_processing_and_logging(accumulator, info_from_response):
            OpenAIWrapper._format_accumulator_in_place(accumulator)

            model = info_from_response.get("model") or llm_configuration.model
            OpenAIWrapper.update_trace_data_from_stream_response(trace_id, model, accumulator, kwargs, info_from_response["first_token_timestamp"])
            final_log()

        if is_old_openai:

            def logging_sync_generator(sync_gen: Generator[_T]) -> Generator[_T]:
                for chunk in sync_gen:
                    self._update_accumulator_streaming(accumulator, info_from_response, chunk)
                    yield chunk
                gen_final_processing_and_logging(accumulator, info_from_response)

            return logging_sync_generator(response)

        else:
            from parea.types import OpenAIStreamWrapper

            return OpenAIStreamWrapper(
                response,
                accumulator,
                info_from_response,
                OpenAIWrapper._update_accumulator_streaming,
                gen_final_processing_and_logging,
            )

    @staticmethod
    def _get_default_dict_streaming():
        info_from_response = {"model": "", "first_token_timestamp": None}
        accumulated_tools = defaultdict(lambda: {"id": "", "function": {"arguments": [], "name": ""}, "type": "function"})
        accumulator = defaultdict(content=[], role="assistant", function_call=defaultdict(arguments=[], name=""), tool_calls=accumulated_tools)
        return accumulator, info_from_response

    @staticmethod
    def _format_accumulator_in_place(accumulator):
        content = "".join(accumulator.get("content"))
        accumulator["content"] = content

        tool_calls = []
        if accumulated_tools := accumulator.get("tool_calls", {}):
            for tool in accumulated_tools.values():
                tool["function"]["arguments"] = "".join(tool["function"]["arguments"])

            tool_calls = list(accumulated_tools.values())
            for tool in tool_calls:
                tool["function"]["arguments"] = json_dumps(json.loads(tool["function"]["arguments"]), indent=2)
        accumulator["tool_calls"] = tool_calls

        if function_call := accumulator.get("function_call"):
            if function_call.get("name"):
                function_call["arguments"] = json_dumps(json.loads("".join(function_call["arguments"])), indent=2)
            else:
                function_call["arguments"] = ""

        return accumulator

    @staticmethod
    def _update_accumulator_streaming(accumulator, info_from_response, chunk):
        def _set_timestamp_if_not_set():
            if not info_from_response.get("first_token_timestamp"):
                info_from_response["first_token_timestamp"] = timezone_aware_now()

        if chunk.model and not info_from_response.get("model"):
            info_from_response["model"] = chunk.model

        if not chunk.choices:
            return

        if is_old_openai:
            delta_dict = chunk.choices[0].delta._previous
        else:
            delta_dict = chunk.choices[0].delta.model_dump()

        if not accumulator["role"]:
            accumulator["role"] = delta_dict.get("role")

        if delta_dict.get("content"):
            _set_timestamp_if_not_set()
            accumulator["content"].append(delta_dict["content"])

        if delta_dict.get("function_call"):
            _set_timestamp_if_not_set()
            accumulator["function_call"]["name"] = delta_dict.get("function_call")["name"] or accumulator["function_call"]["name"]
            if delta_dict.get("function_call", {}).get("arguments"):
                accumulator["function_call"]["arguments"].append(delta_dict["function_call"]["arguments"])

        if tool_calls := delta_dict.get("tool_calls", []):
            _set_timestamp_if_not_set()
            for tool_call in tool_calls:
                tool_id = tool_call["index"]

                accumulator["tool_calls"][tool_id]["id"] = tool_call.get("id") or accumulator["tool_calls"][tool_id]["id"]
                if not accumulator["tool_calls"][tool_id]["type"]:
                    accumulator["tool_calls"][tool_id]["type"] = "function"

                accumulator["tool_calls"][tool_id]["function"]["name"] = tool_call.get("function", {}).get("name") or accumulator["tool_calls"][tool_id]["function"]["name"]
                if tool_call.get("function", {}).get("arguments"):
                    accumulator["tool_calls"][tool_id]["function"]["arguments"].append(tool_call["function"]["arguments"])

    def agen_resolver(self, trace_id: str, _args: Sequence[Any], kwargs: Dict[str, Any], response, final_log):
        llm_configuration = self._kwargs_to_llm_configuration(kwargs)
        trace_data.get()[trace_id].configuration = llm_configuration

        accumulator, info_from_response = self._get_default_dict_streaming()

        def agen_final_processing_and_logging(accumulator, info_from_response):
            OpenAIWrapper._format_accumulator_in_place(accumulator)

            model = info_from_response.get("model") or llm_configuration.model
            OpenAIWrapper.update_trace_data_from_stream_response(trace_id, model, accumulator, kwargs, info_from_response["first_token_timestamp"])
            final_log()

        if is_old_openai:

            async def logging_async_generator(async_gen: AsyncGenerator[_T]) -> AsyncGenerator[_T]:
                async for chunk in async_gen:
                    self._update_accumulator_streaming(accumulator, info_from_response, chunk)
                    yield chunk
                agen_final_processing_and_logging(accumulator, info_from_response)

            return logging_async_generator(response)
        else:
            from parea.types import OpenAIAsyncStreamWrapper

            return OpenAIAsyncStreamWrapper(
                response,
                accumulator,
                info_from_response,
                OpenAIWrapper._update_accumulator_streaming,
                agen_final_processing_and_logging,
            )

    @staticmethod
    def update_trace_data_from_stream_response(trace_id, model, accumulator, kwargs, first_token_timestamp):
        output = OpenAIWrapper._get_output(accumulator, model)
        trace_data.get()[trace_id].output = output
        output_tokens = _num_tokens_from_string(output, model)
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
        trace_data.get()[trace_id].time_to_first_token = (first_token_timestamp - datetime.fromisoformat(trace_data.get()[trace_id].start_timestamp)).total_seconds()

    @staticmethod
    def _kwargs_to_llm_configuration(kwargs):
        return _kwargs_to_llm_configuration(kwargs)

    @staticmethod
    def _get_output(result: Any, model: Optional[str] = None) -> str:
        if not isinstance(result, OpenAIObject) and isinstance(result, dict):
            result = convert_to_openai_object(
                {
                    "choices": [
                        {
                            "index": 0,
                            "message": result,
                        }
                    ],
                    "model": model or "model",
                }
            )
        response_message = result.choices[0].message
        if not response_message.get("content", None) if is_old_openai else not response_message.content:
            completion = OpenAIWrapper._format_function_call(response_message)
        else:
            completion = response_message.content
        return completion

    @staticmethod
    def _format_function_call(response_message) -> str:
        return _format_function_call(response_message)

    def convert_kwargs_to_cache_request(self, _args: Sequence[Any], kwargs: Dict[str, Any]) -> CacheRequest:
        return CacheRequest(
            configuration=self._kwargs_to_llm_configuration(kwargs),
        )

    @staticmethod
    def _convert_cache_to_response(_args: Sequence[Any], kwargs: Dict[str, Any], cache_response: TraceLog) -> OpenAIObject:
        content = cache_response.output
        message = {"role": "assistant", "content": None}
        try:
            function_or_tool_call = json.loads(content)
            if isinstance(function_or_tool_call, list) and len(function_or_tool_call) == 1:
                if "name" in function_or_tool_call[0] and "arguments" in function_or_tool_call[0]:
                    message["function_call"] = function_or_tool_call[0]
            elif isinstance(function_or_tool_call, list) and len(function_or_tool_call) > 1:
                tool_calls = []
                for idx, tool_call in enumerate(function_or_tool_call):
                    if "name" in tool_call and "arguments" in tool_call:
                        tool_calls.append({"id": idx, "function": tool_call, "type": "function"})
                message["tool_calls"] = tool_calls
            else:
                message["content"] = content
        except json.JSONDecodeError:
            message["content"] = content

        message_field = "delta" if kwargs.get("stream", False) else "message"

        return convert_to_openai_object(
            {
                "object": "chat.completion",
                "model": cache_response.configuration.model,
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

    def convert_cache_to_response(self, _args: Sequence[Any], kwargs: Dict[str, Any], cache_response: TraceLog) -> Union[OpenAIObject, Iterator[OpenAIObject]]:
        response = self._convert_cache_to_response(_args, kwargs, cache_response)
        if kwargs.get("stream", False):
            return iter([response])
        else:
            return response

    def aconvert_cache_to_response(self, _args: Sequence[Any], kwargs: Dict[str, Any], cache_response: TraceLog) -> Union[OpenAIObject, AsyncIterator[OpenAIObject]]:
        response = self._convert_cache_to_response(_args, kwargs, cache_response)
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
