import os

import httpx
from dotenv import load_dotenv

from parea import Parea, aprocess_stream_and_yield, convert_openai_raw_to_log, process_stream_and_yield, trace
from parea.cookbook.data.openai_input_examples import functions_example, simple_example, tool_calling_example
from parea.wrapper import get_formatted_openai_response

load_dotenv()

p = Parea(api_key=os.getenv("PAREA_API_KEY"))

URL = "https://api.openai.com/v1/chat/completions"
HEADERS = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}",
}
TIMEOUT = None

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
