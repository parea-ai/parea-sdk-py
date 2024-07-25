from typing import Callable, Dict, List, Optional, Union

import json
import re
import sys
from functools import lru_cache, wraps

import tiktoken
from openai import __version__ as openai_version

from parea.constants import ALL_NON_AZURE_MODELS_INFO, AZURE_MODEL_INFO, TURN_OFF_PAREA_EVAL_LOGGING
from parea.parea_logger import parea_logger
from parea.schemas.log import LLMInputs, Message, ModelParams, Role
from parea.schemas.models import UpdateTraceScenario
from parea.utils.trace_utils import fill_trace_data, get_current_trace_id, log_in_thread, trace_data, trace_insert
from parea.utils.universal_encoder import json_dumps

is_openai_1 = openai_version.startswith("1.")
if is_openai_1:
    from openai.types.chat import ChatCompletion, ChatCompletionMessage


# https://gist.github.com/JettJones/c236494013f22723c1822126df944b12
def skip_decorator_if_func_in_stack_for_evals(*funcs_to_check: Callable) -> Callable:
    def decorator_wrapper(decorator: Callable) -> Callable:
        def new_decorator(self, func: Callable) -> Callable:  # Include self
            @wraps(func)
            def wrapper(*args, **kwargs):
                if not TURN_OFF_PAREA_EVAL_LOGGING:
                    return decorator(self, func)(*args, **kwargs)  # Include self
                frame = sys._getframe().f_back
                caller_names = ""
                while frame:
                    caller_names += frame.f_code.co_name + "|"
                    frame = frame.f_back
                if any(func_to_check.__name__ in caller_names for func_to_check in funcs_to_check):
                    return func(*args, **kwargs)
                return decorator(self, func)(*args, **kwargs)  # Include self

            return wrapper

        return new_decorator

    return decorator_wrapper


def _safe_encode(encoding, text):
    try:
        return len(encoding.encode(text))
    except Exception as e:
        print(f"Error encoding text: {e}")
        return 0


def _num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613", is_azure: bool = False):
    """Return the number of tokens used by a list of messages.
    source: https://cookbook.openai.com/examples/how_to_count_tokens_with_tiktoken
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
    if (
        model
        in {
            "gpt-3.5-turbo",
            "gpt-3.5-turbo-1106",
            "gpt-3.5-turbo-0125",
            "gpt-3.5-turbo-0613",
            "gpt-3.5-turbo-16k-0613",
            "gpt-4-0314",
            "gpt-4-32k-0314",
            "gpt-4-0613",
            "gpt-4-32k-0613",
            "gpt-4-turbo-preview",
            "gpt-4-1106-preview",
            "gpt-4-0125-preview",
        }
        or is_azure
    ):
        tokens_per_message = 3
        tokens_per_name = 1
    elif model == "gpt-3.5-turbo-0301":
        tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
        tokens_per_name = -1  # if there's a name, the role is omitted
    elif "gpt-4" in model:
        print("Warning: gpt-4 may update over time. Returning num tokens assuming gpt-4-0613.")
        return _num_tokens_from_messages(messages, model="gpt-4-0613")
    else:
        print(
            f"""Warning: num_tokens_from_messages() is not implemented for model {model}.
            See https://github.com/openai/openai-python/blob/main/chatml.md for
            information on how messages are converted to tokens. Will use default model"""
        )
        return _num_tokens_from_messages(messages)
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += _safe_encode(encoding, value)
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens


def _num_tokens_from_functions(functions, function_call, model="gpt-3.5-turbo-0613"):
    """Return the number of tokens used by a list of functions. modified from source:
    https://community.openai.com/t/how-to-calculate-the-tokens-when-using-function-call/266573/10 tested:
    https://deepnote.com/workspace/joel-alexander-7514-38647c2d-ba7d-4573-b016-5233b373912e/project/BloomreachDemo
    -66683e80-4414-4b43-ac70-e7eabd3743c1/notebook/token_counting_openai-992f53bba3224221901dcb88befa2051
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")

    num_tokens = 3 if model in ["gpt-3.5-turbo-1106", "gpt-4-1106-preview"] else 0
    for function in functions:
        try:
            function_tokens = _safe_encode(encoding, function.get("name", ""))
            function_tokens += _safe_encode(encoding, function.get("description", ""))
        except Exception as e:
            print(f"Error counting tokens: {e}")
            function_tokens = 0

        if "parameters" in function:
            parameters = function["parameters"]
            if "properties" in parameters:
                for propertiesKey in parameters["properties"]:
                    function_tokens += _safe_encode(encoding, propertiesKey)
                    v = parameters["properties"][propertiesKey]
                    for field in v:
                        if field == "type":
                            function_tokens += 2
                            function_tokens += _safe_encode(encoding, v["type"])
                        elif field == "description":
                            function_tokens += 2
                            function_tokens += _safe_encode(encoding, v["description"])
                        elif field == "enum":
                            function_tokens -= 3
                            for o in v["enum"]:
                                function_tokens += 3
                                function_tokens += _safe_encode(encoding, o)
                        else:
                            print(f"Warning: not supported field {field}")
                function_tokens += 11

        num_tokens += function_tokens

    num_tokens += 10
    function_call_tokens = min(_safe_encode(encoding, "auto") - 1, 0)
    if isinstance(function_call, dict):
        function_call_tokens = min(_safe_encode(encoding, json_dumps(function_call)) - 1, 0)
    return num_tokens + function_call_tokens


def _num_tokens_from_string(string: str, model_name: str = "gpt-3.5-turbo") -> int:
    """Returns the number of tokens in a text string."""
    try:
        encoding = tiktoken.encoding_for_model(model_name)
    except KeyError:
        print(f"Warning: model {model_name} not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
    return _safe_encode(encoding, string)


def _calculate_input_tokens(
    messages: Optional[List[Dict[str, str]]],
    functions: List[Dict[str, str]],
    function_call: Union[str, Dict[str, str]],
    model: str,
) -> int:
    is_azure = model.startswith("azure_") or model in AZURE_MODEL_INFO
    num_function_tokens = _num_tokens_from_functions(functions, function_call, model)
    num_input_tokens = _num_tokens_from_string(json_dumps(messages), model) if model == "gpt-4-vision-preview" else _num_tokens_from_messages(messages, model, is_azure)
    return num_input_tokens + num_function_tokens


def _format_function_call(response_message) -> str:
    def clean_json_string(s):
        """If OpenAI responds with improper newlines and multiple quotes, this will clean it up"""
        return json_dumps(s.replace("'", '"').replace("\\n", "\\\\n"))

    func_obj = response_message.tool_calls
    if response_message.function_call and response_message.function_call.name:
        func_obj = response_message.function_call

    calls = []
    if not isinstance(func_obj, list):
        func_obj = [func_obj]

    for call in func_obj:
        if call:
            is_tool_call = hasattr(call, "id")
            body = getattr(call, "function", None) or call
            function_name = body.name
            try:
                function_args = json.loads(body.arguments)
            except json.decoder.JSONDecodeError:
                try:
                    function_args = json.loads(clean_json_string(body.arguments))
                except Exception:
                    function_args = str(body.arguments)

            if is_tool_call:
                calls.append(
                    {
                        "id": call.id,
                        "type": call.type,
                        "function": {"name": function_name, "arguments": function_args},
                    }
                )
            else:
                calls.append({"name": function_name, "arguments": function_args})
    return json_dumps(calls, indent=4)


def _resolve_functions(kwargs):
    if "functions" in kwargs:
        return kwargs.get("functions", [])
    elif "tools" in kwargs:
        tools = kwargs["tools"]
        if isinstance(tools, list):
            return [d.get("function", {}) for d in tools]

        return []  # it is either a list or Stainless's `NotGiven`

    return []


def _kwargs_to_llm_configuration(kwargs, model=None) -> LLMInputs:
    functions = _resolve_functions(kwargs)
    function_call_default = "auto" if functions else None
    return LLMInputs(
        model=model or kwargs.get("model", None),
        provider="openai",
        messages=_convert_oai_messages(kwargs.get("messages", None)),
        functions=functions,
        function_call=kwargs.get("function_call", function_call_default) or kwargs.get("tool_choice", function_call_default),
        model_params=ModelParams(
            temp=kwargs.get("temperature", 1.0),
            max_length=kwargs.get("max_tokens", None),
            top_p=kwargs.get("top_p", 1.0),
            frequency_penalty=kwargs.get("frequency_penalty", 0.0),
            presence_penalty=kwargs.get("presence_penalty", 0.0),
            response_format=kwargs.get("response_format", None),
        ),
    )


def _convert_oai_messages(messages: list) -> Union[List[Union[dict, Message]], None]:
    if not messages:
        return messages
    if is_openai_1:
        cleaned_messages = []
        for m in messages:
            is_chat_completion = isinstance(m, ChatCompletionMessage)
            if (is_chat_completion and m.role == "tool") or (isinstance(m, dict) and m.get("role") == "tool"):
                tool_call_id = m.tool_call_id if is_chat_completion else m.get("tool_call_id")
                content = m.content if is_chat_completion else m.get("content", "")
                cleaned_messages.append(
                    Message(
                        role=Role.tool,
                        content=json_dumps({"tool_call_id": tool_call_id, "content": content}, indent=4),
                    )
                )
            elif is_chat_completion:
                cleaned_messages.append(
                    Message(
                        role=m.role,
                        content=m.content if m.content else _format_function_call(m),
                    )
                )
            else:
                if "tool_calls" in m:
                    cleaned_messages.append(
                        Message(
                            role=m["role"],
                            content=_format_function_call(ChatCompletionMessage(**m)),
                        )
                    )
                else:
                    cleaned_messages.append(m)
        return cleaned_messages
    else:
        return messages


@lru_cache(maxsize=128)
def _compute_cost(prompt_tokens: int, completion_tokens: int, model: str) -> float:
    if model in AZURE_MODEL_INFO:
        cost_per_token = AZURE_MODEL_INFO[model]
    else:
        cost_per_token = ALL_NON_AZURE_MODELS_INFO.get(model, {"prompt": 0, "completion": 0})
    cost = ((prompt_tokens * cost_per_token["prompt"]) + (completion_tokens * cost_per_token["completion"])) / 1_000_000
    cost = round(cost, 10)
    return cost


def _process_response(response, model_inputs, trace_id):
    response_message = response.choices[0].message
    if response_message.content:
        completion = response_message.content
    else:
        completion = _format_function_call(response_message)

    usage = response.usage
    trace_insert(
        {
            "configuration": _kwargs_to_llm_configuration(model_inputs, response.model),
            "output": completion,
            "input_tokens": usage.prompt_tokens,
            "output_tokens": usage.completion_tokens,
            "total_tokens": usage.prompt_tokens + usage.completion_tokens,
            "cost": _compute_cost(usage.prompt_tokens, usage.completion_tokens, response.model),
        },
        trace_id,
    )


def _process_stream_response(content: list, tools: dict, data: dict, trace_id: str):
    # format to Log schema and send to logger
    model = data.get("model")
    final_content = "".join(content)

    for tool in tools.values():
        tool["function"]["arguments"] = "".join(tool["function"]["arguments"])

    tool_calls = [t["function"] for t in tools.values()]
    for tool in tool_calls:
        try:
            tool["arguments"] = json.loads(tool["arguments"])
        except Exception:
            tool["arguments"] = str(tool["arguments"])

    completion = final_content or json_dumps(tool_calls, indent=4)

    prompt_tokens = _calculate_input_tokens(
        data.get("messages", []),
        data.get("functions", []) or [d["function"] for d in data.get("tools", [])],
        data.get("function_call", "auto") or data.get("tool_choice", "auto"),
        data.get("model"),
    )
    completion_tokens = _num_tokens_from_string(final_content if final_content else json_dumps(tool_calls), model)
    data = {
        "configuration": _kwargs_to_llm_configuration(data, model),
        "output": completion,
        "input_tokens": prompt_tokens,
        "output_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "cost": _compute_cost(prompt_tokens, completion_tokens, model),
    }
    fill_trace_data(trace_id, data, UpdateTraceScenario.OPENAICONFIG)
    data_with_config = trace_data.get()[trace_id]
    parea_logger.default_log(data=data_with_config)


def convert_openai_raw_stream_to_log(content: list, tools: dict, data: dict, trace_id: str):
    log_in_thread(
        _process_stream_response,
        {"content": content, "tools": tools, "data": data, "trace_id": trace_id},
    )


def convert_openai_raw_to_log(r: dict, data: dict):
    log_in_thread(
        _process_response,
        {
            "response": ChatCompletion(**r),
            "model_inputs": data,
            "trace_id": get_current_trace_id(),
        },
    )


def safe_format_template_to_prompt(_template: str, **kwargs) -> str:
    """Replaces langchain.prompts.PromptTemplate.format in a safe manner.

    Only variables '{{...}}' will be replaced, not any in '{...}'
    """

    def replace(match):
        var_name = match.group(1)
        return str(kwargs.get(var_name, match.group(0)))

    return re.sub(r"{{(\w+)}}", replace, _template)
