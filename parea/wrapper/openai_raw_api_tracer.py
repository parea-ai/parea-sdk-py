from typing import Any, AsyncGenerator, Generator

import json
from collections import defaultdict

from parea.constants import CHUNK_DONE_SENTINEL
from parea.utils.trace_utils import get_current_trace_id
from parea.utils.universal_encoder import json_dumps
from parea.wrapper.utils import convert_openai_raw_stream_to_log


def process_stream_and_yield(response, data: dict) -> Generator:
    trace_id = get_current_trace_id()
    accumulated_content = []
    accumulated_tools = defaultdict(lambda: {"function": {"arguments": [], "name": ""}})

    for chunk in response.iter_lines():
        format_and_accumulate_streaming_chunk(trace_id, accumulated_content, accumulated_tools, data, chunk)
        yield chunk


async def aprocess_stream_and_yield(response, data: dict) -> AsyncGenerator:
    trace_id = get_current_trace_id()
    accumulated_content = []
    accumulated_tools = defaultdict(lambda: {"function": {"arguments": [], "name": ""}})

    async for chunk in response.aiter_lines():
        format_and_accumulate_streaming_chunk(trace_id, accumulated_content, accumulated_tools, data, chunk)
        yield chunk


def format_and_accumulate_streaming_chunk(trace_id: str, accumulated_content: list, accumulated_tools: dict, data: dict, chunk: Any) -> None:
    from openai.types.chat import ChatCompletionChunk

    try:
        chunk = chunk.decode("utf-8")
    except AttributeError:
        pass
    if chunk == CHUNK_DONE_SENTINEL:
        # when done send accumulated content to be logged in background thread
        convert_openai_raw_stream_to_log(accumulated_content, accumulated_tools, data, trace_id)
    else:
        chunk_data = raw_chunk_to_chat_completion_chunk(chunk)
        if isinstance(chunk_data, ChatCompletionChunk):
            for choice in chunk_data.choices or []:
                delta = choice.delta

                if delta.content:
                    accumulated_content.append(delta.content)

                if delta.function_call:
                    accumulated_tools[0]["function"]["name"] = delta.function_call.name or accumulated_tools[0]["function"]["name"]
                    if delta.function_call.arguments:
                        accumulated_tools[0]["function"]["arguments"].append(delta.function_call.arguments)

                for tool_call in delta.tool_calls or []:
                    tool_id = tool_call.index
                    accumulated_tools[tool_id]["function"]["name"] = tool_call.function.name or accumulated_tools[tool_id]["function"]["name"]
                    if tool_call.function.arguments:
                        accumulated_tools[tool_id]["function"]["arguments"].append(tool_call.function.arguments)


def raw_chunk_to_chat_completion_chunk(chunk: str):
    from openai.types.chat import ChatCompletionChunk

    try:
        return ChatCompletionChunk(**json.loads(chunk[6:].strip()))
    except json.JSONDecodeError:
        return chunk


def get_formatted_openai_response(r):
    # helper function to format the response from OpenAI
    if r["choices"][0]["message"].get("content"):
        return r["choices"][0]["message"]["content"].strip()
    elif r["choices"][0]["message"].get("function_call"):
        function_call = r["choices"][0]["message"]["function_call"]
        formatted_function_call = {
            "name": function_call["name"],
            "arguments": json.loads(function_call["arguments"]),
        }
        return json_dumps(formatted_function_call, indent=4)
    elif r["choices"][0]["message"].get("tool_calls"):
        formatted_tool_calls = []
        tool_calls = r["choices"][0]["message"]["tool_calls"]
        for tool_call in tool_calls:
            formatted_tool_call = {
                "name": tool_call["function"]["name"],
                "arguments": json.loads(tool_call["function"]["arguments"]),
            }
            formatted_tool_calls.append(formatted_tool_call)
        return json_dumps(formatted_tool_calls, indent=4)
    return json_dumps(r, indent=4)
