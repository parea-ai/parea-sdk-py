import os

import httpx
from dotenv import load_dotenv

from parea import Parea
from parea.utils.trace_utils import trace
from parea.wrapper.openai_raw_api_tracer import aprocess_stream_and_yield, get_formatted_openai_response, process_stream_and_yield
from parea.wrapper.utils import convert_openai_raw_to_log

load_dotenv()

p = Parea(api_key="pai-83bc9a58b025773fae7bc1e8b2518d32c4c5862f70116a1124021af0e46d046f")

URL = "https://api.openai.com/v1/chat/completions"
HEADERS = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}",
}
TIMEOUT = None

tool_calling_example = {
    "model": "gpt-3.5-turbo-1106",
    "messages": [{"role": "user", "content": "What's the weather like in San Francisco, Tokyo, and Paris?"}],
    "tools": [
        {
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "description": "Get the current weather in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA",
                        },
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                    },
                    "required": ["location"],
                },
            },
        }
    ],
    "tool_choice": "auto",
}

functions_example = {
    "model": "gpt-3.5-turbo-1106",
    "messages": [
        {
            "role": "system",
            "content": f"You are a sophisticated AI assistant, "
            f"a specialist in user intent detection and interpretation. "
            f"Your task is to perceive and respond to the user's needs, even when they're expressed "
            f"in an indirect or direct manner. You excel in recognizing subtle cues: for example, "
            f"if a user states they are 'hungry', you should assume they are seeking nearby dining "
            f"options such as a restaurant or a cafe. If they indicate feeling 'tired', 'weary', "
            f"or mention a long journey, interpret this as a request for accommodation options like "
            f"hotels or guest houses. However, remember to navigate the fine line of interpretation "
            f"and assumption: if a user's intent is unclear or can be interpreted in multiple ways, "
            f"do not hesitate to politely ask for additional clarification. Make sure to tailor your "
            f"responses to the user based on their preferences and past experiences which can "
            f"be found here: Name: John Doe",
        },
        {"role": "user", "content": "I'm hungry"},
    ],
    "functions": [
        {
            "name": "call_google_places_api",
            "description": f"""
            This function calls the Google Places API to find the top places of a specified type near
            a specific location. It can be used when a user expresses a need (e.g., feeling hungry or tired) or wants to
            find a certain type of place (e.g., restaurant or hotel).
        """,
            "parameters": {"type": "object", "properties": {"place_type": {"type": "string", "description": "The type of place to search for."}}},
            "result": {"type": "array", "items": {"type": "string"}},
        }
    ],
}

simple_example = {"model": "gpt-3.5-turbo-1106", "messages": [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "Hello!"}]}


# Sync HTTPX


## Normal
@trace
def call_openai_api(data: dict) -> str:
    with httpx.Client(timeout=TIMEOUT) as client:
        response = client.post(URL, json=data, headers=HEADERS)
        r = response.json()
        convert_openai_raw_to_log(r, data)  # Add this line to enable tracing. Non-blocking
        return get_formatted_openai_response(r)  # Return how you normally would


## Streaming
@trace
def call_openai_api_stream(data: dict):
    data["stream"] = True
    with httpx.stream("POST", URL, json=data, headers=HEADERS, timeout=TIMEOUT) as response:
        # Add process_stream_and_yield to enable tracing. Non-blocking
        for chunk in process_stream_and_yield(response, data):
            print(chunk)


# Async HTTPX


## Normal
@trace
async def acall_openai_api(data: dict) -> str:
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        response = await client.post(URL, json=data, headers=HEADERS)
        r = response.json()
        convert_openai_raw_to_log(r, data)  # Add this line to enable tracing. Non-blocking
        return get_formatted_openai_response(r)  # Return how you normally would


## Streaming
@trace
async def acall_openai_api_stream(data: dict):
    data["stream"] = True
    async with httpx.AsyncClient(timeout=TIMEOUT).stream("POST", URL, json=data, headers=HEADERS) as response:
        # Add process_stream_and_yield to enable tracing. Non-blocking
        async for chunk in aprocess_stream_and_yield(response, data):
            print(chunk)


# TEST NESTED TRACING
@trace
def chain():
    call_openai_api(simple_example)
    call_openai_api(functions_example)
    call_openai_api(tool_calling_example)
    call_openai_api_stream(tool_calling_example)


@trace
async def achain():
    await acall_openai_api(simple_example)
    await acall_openai_api(functions_example)
    await acall_openai_api(tool_calling_example)
    await acall_openai_api_stream(tool_calling_example)


if __name__ == "__main__":
    chain()
    # asyncio.run(achain())
