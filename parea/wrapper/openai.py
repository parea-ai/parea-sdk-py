from typing import Any, AsyncIterator, Callable, Dict, Iterator, Optional, Sequence, Union

import json
from collections import defaultdict

import openai
from openai.openai_object import OpenAIObject
from openai.util import convert_to_openai_object

from ..cache.cache import Cache
from ..schemas.models import CacheRequest, LLMInputs, ModelParams, TraceLog
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

    def init(self, log: Callable, cache: Cache = None):
        Wrapper(
            resolver=self.resolver,
            gen_resolver=self.gen_resolver,
            agen_resolver=self.agen_resolver,
            log=log,
            module=openai,
            func_names=list(self.original_methods.keys()),
            cache=cache,
            convert_kwargs_to_cache_request=self.convert_kwargs_to_cache_request,
            convert_cache_to_response=self.convert_cache_to_response,
            aconvert_cache_to_response=self.aconvert_cache_to_response,
        )

    @staticmethod
    def resolver(trace_id: str, _args: Sequence[Any], kwargs: Dict[str, Any], response: Optional[Any]) -> Optional[Any]:
        if response:
            usage = response["usage"]
            output = OpenAIWrapper._get_output(response)
        else:
            output = None
            usage = {}

        llm_configuration = OpenAIWrapper._kwargs_to_llm_configuration(kwargs)
        model = llm_configuration.model

        model_rate = OpenAIWrapper.get_model_cost(model)
        model_completion_rate = OpenAIWrapper.get_model_cost(model, is_completion=True)
        completion_cost = model_completion_rate * (usage.get("completion_tokens", 0) / 1000)
        prompt_cost = model_rate * (usage.get("prompt_tokens", 0) / 1000)
        total_cost = sum([prompt_cost, completion_cost])

        trace_data.get()[trace_id].configuration = llm_configuration
        trace_data.get()[trace_id].input_tokens = usage.get("prompt_tokens", 0)
        trace_data.get()[trace_id].output_tokens = usage.get("completion_tokens", 0)
        trace_data.get()[trace_id].total_tokens = usage.get("total_tokens", 0)
        trace_data.get()[trace_id].cost = total_cost
        trace_data.get()[trace_id].output = output
        return response

    @staticmethod
    def gen_resolver(trace_id: str, _args: Sequence[Any], kwargs: Dict[str, Any], response: Iterator[Any], final_log) -> Iterator[Any]:
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
    async def agen_resolver(trace_id: str, _args: Sequence[Any], kwargs: Dict[str, Any], response: AsyncIterator[Any], final_log) -> AsyncIterator[Any]:
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
        if response_message.get("function_call", None):
            completion = OpenAIWrapper._format_function_call(response_message)
        else:
            completion = response_message.content.strip()
        return completion

    @staticmethod
    def _format_function_call(response_message) -> str:
        function_name = response_message["function_call"]["name"]
        if isinstance(response_message["function_call"]["arguments"], OpenAIObject):
            function_args = dict(response_message["function_call"]["arguments"])
        else:
            function_args = json.loads(response_message["function_call"]["arguments"])
        return json.dumps({"name": function_name, "arguments": function_args}, indent=4)

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

    @staticmethod
    def convert_kwargs_to_cache_request(_args: Sequence[Any], kwargs: Dict[str, Any]) -> CacheRequest:
        return CacheRequest(
            configuration=OpenAIWrapper._kwargs_to_llm_configuration(kwargs),
        )

    @staticmethod
    def _convert_cache_to_response(_args: Sequence[Any], kwargs: Dict[str, Any], cache_response: TraceLog) -> OpenAIObject:
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
    def convert_cache_to_response(_args: Sequence[Any], kwargs: Dict[str, Any], cache_response: TraceLog) -> Union[OpenAIObject, Iterator[OpenAIObject]]:
        response = OpenAIWrapper._convert_cache_to_response(_args, kwargs, cache_response)
        if kwargs.get("stream", False):
            return iter([response])
        else:
            return response

    @staticmethod
    def aconvert_cache_to_response(_args: Sequence[Any], kwargs: Dict[str, Any], cache_response: TraceLog) -> Union[OpenAIObject, AsyncIterator[OpenAIObject]]:
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
