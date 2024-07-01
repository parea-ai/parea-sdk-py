from typing import Any, Callable, Dict, Optional, Sequence

from collections import defaultdict
from copy import deepcopy
from datetime import datetime

from anthropic import AsyncMessageStreamManager, AsyncStream, Client, MessageStreamManager, Stream
from anthropic.types import ContentBlockDeltaEvent, Message, MessageDeltaEvent, MessageStartEvent, TextBlock

from parea.cache.cache import Cache
from parea.helpers import timezone_aware_now
from parea.schemas import CacheRequest, LLMInputs, ModelParams
from parea.schemas import Role as PareaRole
from parea.schemas import TraceLog
from parea.utils.trace_utils import make_output, trace_data
from parea.wrapper import Wrapper
from parea.wrapper.anthropic.stream_wrapper import AnthropicAsyncStreamWrapper, AnthropicStreamWrapper, MessageAsyncStreamManagerWrapper, MessageStreamManagerWrapper
from parea.wrapper.utils import _compute_cost


class AnthropicWrapper:

    def init(self, log: Callable, cache: Cache, client: Client):
        func_names = ["messages.create", "messages.stream"]
        if hasattr(client, "beta") and hasattr(client.beta, "tools") and hasattr(client.beta.tools, "messages") and hasattr(client.beta.tools.messages, "create"):
            func_names.append("beta.tools.messages.create")
        Wrapper(
            resolver=self.resolver,
            gen_resolver=self.gen_resolver,
            agen_resolver=self.agen_resolver,
            should_use_gen_resolver=self.should_use_gen_resolver,
            log=log,
            module=client,
            func_names=func_names,
            cache=cache,
            convert_kwargs_to_cache_request=self.convert_kwargs_to_cache_request,
            convert_cache_to_response=self.convert_cache_to_response,
            aconvert_cache_to_response=self.aconvert_cache_to_response,
            name="llm-anthropic",
        )

    @staticmethod
    def resolver(trace_id: str, _args: Sequence[Any], kwargs: Dict[str, Any], response: Optional[Message]) -> Optional[Any]:
        if response:
            if len(response.content) > 1:
                from anthropic.types.beta.tools import ToolUseBlock

                output_list = []
                for content in response.content:
                    if isinstance(content, TextBlock):
                        output_list.append(content.text)
                    elif isinstance(content, ToolUseBlock):
                        output_list.append(content.model_dump())
                output = make_output(output_list, islist=True)
            else:
                output = response.content[0].text
            input_tokens = response.usage.input_tokens
            output_tokens = response.usage.output_tokens
            total_tokens = input_tokens + output_tokens
        else:
            output = None
            input_tokens, output_tokens, total_tokens = 0, 0, 0

        llm_configuration = AnthropicWrapper._kwargs_to_llm_configuration(kwargs)
        model = response.model if response and response.model else llm_configuration.model

        trace_data.get()[trace_id].configuration = llm_configuration
        trace_data.get()[trace_id].input_tokens = input_tokens
        trace_data.get()[trace_id].output_tokens = output_tokens
        trace_data.get()[trace_id].total_tokens = total_tokens
        trace_data.get()[trace_id].cost = _compute_cost(input_tokens, output_tokens, model)
        trace_data.get()[trace_id].output = output
        return response

    def gen_resolver(self, trace_id: str, _args: Sequence[Any], kwargs: Dict[str, Any], response, final_log):
        if isinstance(response, Stream):
            llm_configuration = self._kwargs_to_llm_configuration(kwargs)
            trace_data.get()[trace_id].configuration = llm_configuration
            accumulator, info_from_response = self._get_default_dict_streaming()

            def gen_final_processing_and_logging(accumulator, info_from_response):
                model = info_from_response.get("model") or llm_configuration.model
                self.update_trace_data_from_stream_response(trace_id, model, accumulator, info_from_response)
                final_log()

            return AnthropicStreamWrapper(
                response,
                accumulator,
                info_from_response,
                self._update_accumulator_streaming,
                gen_final_processing_and_logging,
            )
        else:  # needed for stream context manager; this will need to handle both sync and async; as messages.stream is always sync

            def resolve_and_log(m: Message):
                self.resolver(trace_id, _args, kwargs, m)
                final_log()

            if isinstance(response, MessageStreamManager):
                return MessageStreamManagerWrapper(response, resolve_and_log)
            else:
                return MessageAsyncStreamManagerWrapper(response, resolve_and_log)

    def agen_resolver(self, trace_id: str, _args: Sequence[Any], kwargs: Dict[str, Any], response, final_log):
        llm_configuration = self._kwargs_to_llm_configuration(kwargs)
        trace_data.get()[trace_id].configuration = llm_configuration
        accumulator, info_from_response = self._get_default_dict_streaming()

        def gen_final_processing_and_logging(accumulator, info_from_response):
            model = info_from_response.get("model") or llm_configuration.model
            self.update_trace_data_from_stream_response(trace_id, model, accumulator, info_from_response)
            final_log()

        return AnthropicAsyncStreamWrapper(
            response,
            accumulator,
            info_from_response,
            self._update_accumulator_streaming,
            gen_final_processing_and_logging,
        )

    @staticmethod
    def _get_default_dict_streaming():
        info_from_response = {"model": "", "first_token_timestamp": None, "input_tokens": None, "output_tokens": None}
        accumulator = defaultdict(content=[], role="assistant")
        return accumulator, info_from_response

    @staticmethod
    def _kwargs_to_llm_configuration(kwargs, model=None) -> LLMInputs:
        functions = deepcopy([d for d in kwargs.get("tools", [])])
        messages = kwargs.get("messages", None)
        if system_msg := kwargs.get("system", None):
            if not messages:
                messages = []
            messages = messages.copy()
            messages.insert(0, dict(role=PareaRole.system, content=system_msg))
        for func in functions:
            if "input_schema" in func:
                func["parameters"] = func.pop("input_schema")
        return LLMInputs(
            model=model or kwargs.get("model", None),
            provider="anthropic",
            messages=messages,
            model_params=ModelParams(
                temp=kwargs.get("temperature", 1.0),
                max_length=kwargs.get("max_tokens", None),
                top_p=kwargs.get("top_p", 1.0),
            ),
            functions=functions,
        )

    def convert_kwargs_to_cache_request(self, _args: Sequence[Any], kwargs: Dict[str, Any]) -> CacheRequest:
        pass

    @staticmethod
    def update_trace_data_from_stream_response(trace_id, model, accumulator, model_info):
        output = accumulator["content"]
        if isinstance(output, list):
            output = "".join(output)
        trace_data.get()[trace_id].output = output
        output_tokens = model_info.get("output_tokens", 0)
        input_tokens = model_info.get("input_tokens", 0)
        trace_data.get()[trace_id].input_tokens = input_tokens
        trace_data.get()[trace_id].output_tokens = output_tokens
        trace_data.get()[trace_id].total_tokens = input_tokens + output_tokens
        trace_data.get()[trace_id].cost = _compute_cost(input_tokens, output_tokens, model)
        trace_data.get()[trace_id].time_to_first_token = (model_info["first_token_timestamp"] - datetime.fromisoformat(trace_data.get()[trace_id].start_timestamp)).total_seconds()

    @staticmethod
    def _convert_cache_to_response(_args: Sequence[Any], kwargs: Dict[str, Any], cache_response: TraceLog):
        pass

    @staticmethod
    def should_use_gen_resolver(response: Any) -> bool:
        return isinstance(response, (Stream, MessageStreamManager, AsyncStream, AsyncMessageStreamManager))

    def convert_cache_to_response(self, _args: Sequence[Any], kwargs: Dict[str, Any], cache_response: TraceLog):
        pass

    def aconvert_cache_to_response(self, _args: Sequence[Any], kwargs: Dict[str, Any], cache_response: TraceLog):
        pass

    @staticmethod
    def _update_accumulator_streaming(accumulator, info_from_response, chunk):
        if isinstance(chunk, MessageStartEvent):
            info_from_response["input_tokens"] = chunk.message.usage.input_tokens
        elif isinstance(chunk, ContentBlockDeltaEvent):
            accumulator["content"].append(chunk.delta.text)
            if not info_from_response.get("first_token_timestamp"):
                info_from_response["first_token_timestamp"] = timezone_aware_now()
        elif isinstance(chunk, MessageDeltaEvent):
            info_from_response["output_tokens"] = chunk.usage.output_tokens
