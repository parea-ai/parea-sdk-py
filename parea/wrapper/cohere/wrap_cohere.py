from typing import Any, Callable, Optional, Tuple, Union

import asyncio
import contextvars
import datetime
import functools
import inspect
import json
import logging
import os
import traceback

import cohere
from attrs import asdict, define
from cohere import ApiMetaBilledUnits, NonStreamedChatResponse, RerankResponse

from parea.constants import COHERE_MODEL_INFO, COHERE_SEARCH_MODELS, PAREA_OS_ENV_EXPERIMENT_UUID
from parea.helpers import gen_trace_id, is_logging_disabled, timezone_aware_now
from parea.schemas import LLMInputs, Message, ModelParams, Role, TraceLog, UpdateTraceScenario
from parea.utils.trace_utils import execution_order_counters, fill_trace_data, logger_record_log, trace_context, trace_data
from parea.utils.universal_encoder import json_dumps

logger = logging.getLogger()

DEFAULT_MODEL = "command-r-plus"
DEFAULT_TEMPERATURE = 0.3
DEFAULT_P = 0.75


@define
class CohereOutput:
    text: Optional[str] = None
    citations: Optional[str] = None
    documents: Optional[str] = None
    search_queries: Optional[str] = None
    search_results: Optional[str] = None


class CohereClientWrapper:
    @staticmethod
    def wrap_method(method: Callable, method_name: str) -> Callable:
        """
        Wrap a method with logging functionality based on its type.

        :param method: The original method to be wrapped
        :param method_name: The name of the method for logging purposes
        :return: Wrapped method
        """

        @functools.wraps(method)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            if is_logging_disabled():
                return method(*args, **kwargs)

            trace_id, start_time, context_token = CohereClientWrapper.init_trace(method_name, args, kwargs, method)
            try:
                if method_name == "chat_stream":
                    return CohereClientWrapper._handle_stream(trace_id, start_time, context_token, method, *args, **kwargs)

                result = method(*args, **kwargs)
                if not is_logging_disabled():
                    CohereClientWrapper._fill_llm_config(trace_id, result, **kwargs)
                return result
            except Exception as e:
                logger.error(f"Error occurred in function {method.__name__}, {e}", exc_info=True)
                fill_trace_data(trace_id, {"error": traceback.format_exc()}, UpdateTraceScenario.ERROR)
                raise e
            finally:
                try:
                    if method_name != "chat_stream":
                        CohereClientWrapper.cleanup_trace(trace_id, start_time, context_token)
                except Exception as e:
                    logger.debug(f"Error occurred cleaning up trace for function {method.__name__}, {e}", exc_info=e)

        @functools.wraps(method)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            if is_logging_disabled():
                return await method(*args, **kwargs)

            trace_id, start_time, context_token = CohereClientWrapper.init_trace(method_name, args, kwargs, method)
            try:
                result = await method(*args, **kwargs)
                if not is_logging_disabled():
                    CohereClientWrapper._fill_llm_config(trace_id, result, **kwargs)
                return result
            except Exception as e:
                logger.error(f"Error occurred in function {method.__name__}, {e}")
                fill_trace_data(trace_id, {"error": traceback.format_exc()}, UpdateTraceScenario.ERROR)
                raise e
            finally:
                try:
                    if method_name != "chat_stream":
                        CohereClientWrapper.cleanup_trace(trace_id, start_time, context_token)
                except Exception as e:
                    logger.debug(f"Error occurred cleaning up trace for function {method.__name__}, {e}", exc_info=e)

        if asyncio.iscoroutinefunction(method):
            return async_wrapper
        else:
            return sync_wrapper

    @staticmethod
    def init_trace(func_name, args, kwargs, func) -> Tuple[str, datetime, contextvars.Token]:
        start_time = timezone_aware_now()
        trace_id = gen_trace_id()

        new_trace_context = trace_context.get() + [trace_id]
        token = trace_context.set(new_trace_context)

        try:
            inputs = None
            if kwargs.get("query", None):
                sig = inspect.signature(func)
                parameters = sig.parameters

                inputs = {k: v for k, v in zip(parameters.keys(), args)}
                inputs.update(kwargs)

                # filter out any values which aren't JSON serializable
                for k, v in inputs.items():
                    try:
                        json.dumps(v)
                    except TypeError:
                        try:
                            inputs[k] = json_dumps(v)
                        except (TypeError, AttributeError):
                            # if we can't serialize the value, just convert it to a string
                            inputs[k] = str(v)

            depth = len(new_trace_context) - 1
            root_trace_id = new_trace_context[0]

            # Get the execution order counter for the current root trace
            counters = execution_order_counters.get()
            if root_trace_id not in counters:
                counters[root_trace_id] = 0
            execution_order = counters[root_trace_id]
            counters[root_trace_id] += 1

            trace_data.get()[trace_id] = TraceLog(
                trace_id=trace_id,
                parent_trace_id=root_trace_id,
                root_trace_id=root_trace_id,
                start_timestamp=start_time.isoformat(),
                trace_name="llm-cohere",
                inputs=inputs,
                experiment_uuid=os.environ.get(PAREA_OS_ENV_EXPERIMENT_UUID, None),
                depth=depth,
                execution_order=execution_order,
            )
            parent_trace_id = new_trace_context[-2] if len(new_trace_context) > 1 else None
            if parent_trace_id:
                fill_trace_data(trace_id, {"parent_trace_id": parent_trace_id}, UpdateTraceScenario.CHAIN)
        except Exception as e:
            logger.debug(f"Error occurred initializing trace for function {func_name}, {e}")

        return trace_id, start_time, token

    @staticmethod
    def cleanup_trace(trace_id: str, start_time: datetime, context_token: contextvars.Token, time_to_first_token: Optional[float] = None) -> None:
        end_time = timezone_aware_now()
        trace_data.get()[trace_id].end_timestamp = end_time.isoformat()
        trace_data.get()[trace_id].latency = (end_time - start_time).total_seconds()
        if time_to_first_token:
            trace_data.get()[trace_id].time_to_first_token = time_to_first_token
        logger_record_log(trace_id)
        trace_context.reset(context_token)

    @staticmethod
    def _handle_stream(trace_id: str, start_time: datetime, context_token: contextvars.Token, method: Callable, *args: Any, **kwargs: Any):
        accumulated_text = ""
        final_response: Optional[NonStreamedChatResponse] = None
        time_to_first_token = None

        try:
            for event in method(*args, **kwargs):
                if not time_to_first_token:
                    time_to_first_token = (timezone_aware_now() - start_time).total_seconds()
                if event.event_type == "text-generation":
                    accumulated_text += event.text
                elif event.event_type == "stream-end":
                    final_response = event.response

                yield event
        except Exception as e:
            logger.error(f"Error occurred in function {method.__name__}, {e}")
            fill_trace_data(trace_id, {"error": traceback.format_exc()}, UpdateTraceScenario.ERROR)
            raise e
        finally:
            CohereClientWrapper._fill_llm_config(trace_id, final_response, **kwargs)
            CohereClientWrapper.cleanup_trace(trace_id, start_time, context_token, time_to_first_token)

    @staticmethod
    def _fill_llm_config(trace_id: str, result: Optional[NonStreamedChatResponse | RerankResponse], **kwargs: dict[str, Any]) -> None:
        """
        Fill the LLM configuration data for the given trace.

        :param trace_id: The ID of the current trace
        :param result: The response from the Cohere API
        :param kwargs: Additional keyword arguments
        """
        try:
            model = kwargs.get("model", DEFAULT_MODEL)
            tools = kwargs.get("tools")
            configuration = LLMInputs(
                model=model,
                provider="cohere",
                model_params=ModelParams(
                    temp=kwargs.get("temperature", DEFAULT_TEMPERATURE),
                    top_p=kwargs.get("p", DEFAULT_P),
                    frequency_penalty=kwargs.get("frequency_penalty", 0),
                    presence_penalty=kwargs.get("presence_penalty", 0),
                    max_length=kwargs.get("max_tokens"),
                    response_format=kwargs.get("response_format"),
                ),
                messages=CohereClientWrapper._chat_history_to_messages(result, **kwargs) if isinstance(result, NonStreamedChatResponse) else None,
                functions=json_dumps(tools) if tools else None,
            )
            prompt_tokens, completion_tokens, cost = CohereClientWrapper._get_usage_stats(result, model)
            data = {
                "configuration": configuration,
                "output": CohereClientWrapper._get_output(result),
                "input_tokens": prompt_tokens,
                "output_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
                "cost": cost,
            }
            fill_trace_data(trace_id, data, UpdateTraceScenario.OPENAICONFIG)
        except Exception as e:
            logger.debug(f"Error occurred filling LLM config for trace {trace_id}, {e}", exc_info=True)
            fill_trace_data(trace_id, {"error": traceback.format_exc()}, UpdateTraceScenario.ERROR)

    @staticmethod
    def _chat_history_to_messages(result: NonStreamedChatResponse, **kwargs) -> list[Message]:
        messages: list[Message] = []
        if sys_message := kwargs.get("preamble", ""):
            messages.append(Message(content=sys_message, role=Role.system))
        if history := kwargs.get("chat_history", []):
            messages.extend(CohereClientWrapper._to_messages(history))

        messages.extend(CohereClientWrapper._to_messages([m.dict() for m in result.chat_history]))
        return messages

    @staticmethod
    def _to_messages(chat_history: list[dict]) -> list[Message]:
        messages: list[Message] = []
        for message in chat_history:
            if message["role"] == "USER":
                messages.append(Message(content=message["message"], role=Role.user))
            elif message["role"] == "CHATBOT":
                messages.append(Message(content=message["message"], role=Role.assistant))
            elif message["role"] == "SYSTEM":
                messages.append(Message(content=message["message"], role=Role.system))
            elif message["role"] == "TOOL":
                messages.append(Message(content=json_dumps(message["tool_calls"]), role=Role.tool))

        return messages

    @staticmethod
    @functools.lru_cache(maxsize=128)
    def _compute_cost(prompt_tokens: int, completion_tokens: int, search_units: int, is_search_model: bool, model: str) -> float:
        cost_per_token = COHERE_MODEL_INFO.get(model, {"prompt": 0, "completion": 0})
        cost = ((prompt_tokens * cost_per_token["prompt"]) + (completion_tokens * cost_per_token["completion"])) / 1_000_000
        if is_search_model:
            cost += search_units * cost_per_token.get("search", 0) / 1_000
        cost = round(cost, 10)
        return cost

    @staticmethod
    def _get_usage_stats(result: Optional[NonStreamedChatResponse | RerankResponse], model: str) -> Tuple[int, int, float]:
        bu: Optional[ApiMetaBilledUnits] = result.meta.billed_units if result else None
        if not bu:
            return 0, 0, 0.0
        prompt_tokens = bu.input_tokens or 0
        completion_tokens = bu.output_tokens or 0
        search_units = bu.search_units or 0
        is_search_model: bool = model in COHERE_SEARCH_MODELS
        cost = CohereClientWrapper._compute_cost(prompt_tokens, completion_tokens, search_units, is_search_model, model)
        return prompt_tokens, completion_tokens, cost

    @staticmethod
    def _get_output(result: Optional[NonStreamedChatResponse | RerankResponse]) -> str:
        if not result:
            return ""

        if isinstance(result, RerankResponse):
            output = CohereOutput(documents=CohereClientWrapper._cohere_json_list(result.results) if result.results else None)
            return json_dumps(asdict(output))

        text = result.text or CohereClientWrapper._cohere_json_list(result.tool_calls)
        output = CohereOutput(
            text=text,
            citations=CohereClientWrapper._cohere_json_list(result.citations) if result.citations else None,
            documents=CohereClientWrapper._cohere_json_list(result.documents) if result.documents else None,
            search_queries=CohereClientWrapper._cohere_json_list(result.search_queries) if result.search_queries else None,
            search_results=CohereClientWrapper._cohere_json_list(result.search_results) if result.search_results else None,
        )
        return json_dumps(asdict(output))

    @staticmethod
    def _cohere_json_list(obj: Any) -> str:
        return json_dumps([o.dict() for o in obj])

    @staticmethod
    def init(client: Union[cohere.Client, cohere.AsyncClient]) -> None:
        """
        Apply the CohereClientWrapper to the 'chat', 'chat_stream', and 'rerank' methods of the cohere.Client instance.

        :param client: An instance of cohere.Client
        """
        methods_to_wrap = ["chat", "chat_stream", "rerank"]
        for method_name in methods_to_wrap:
            if hasattr(client, method_name):
                original_method = getattr(client, method_name)
                wrapped_method = CohereClientWrapper.wrap_method(original_method, method_name)
                setattr(client, method_name, wrapped_method)
