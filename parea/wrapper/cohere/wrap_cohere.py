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
from cohere import NonStreamedChatResponse, RerankResponse

from parea.constants import PAREA_OS_ENV_EXPERIMENT_UUID
from parea.helpers import gen_trace_id, is_logging_disabled, timezone_aware_now
from parea.schemas import LLMInputs, ModelParams, TraceLog, UpdateTraceScenario
from parea.utils.trace_utils import execution_order_counters, fill_trace_data, logger_record_log, trace_context, trace_data
from parea.utils.universal_encoder import json_dumps
from parea.wrapper.cohere.helpers import DEFAULT_MODEL, DEFAULT_P, DEFAULT_TEMPERATURE, chat_history_to_messages, get_output, get_usage_stats

logger = logging.getLogger()


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
            tools = kwargs.get("tools", None)
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
                messages=chat_history_to_messages(result, **kwargs) if isinstance(result, NonStreamedChatResponse) else None,
                functions=tools,
            )
            prompt_tokens, completion_tokens, cost = get_usage_stats(result, model)
            data = {
                "configuration": configuration,
                "output": get_output(result),
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
