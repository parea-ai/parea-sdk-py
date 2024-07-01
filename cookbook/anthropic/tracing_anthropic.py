import asyncio
import os

import anthropic
from anthropic.types import ContentBlockDeltaEvent, MessageDeltaEvent, MessageStartEvent
from dotenv import load_dotenv

from parea import Parea

load_dotenv()


client = anthropic.Anthropic()
aclient = anthropic.AsyncAnthropic()

p = Parea(api_key=os.getenv("PAREA_API_KEY"))
p.wrap_anthropic_client(client)
p.wrap_anthropic_client(aclient)


client_kwargs = {"model": "claude-3-opus-20240229", "max_tokens": 1024, "messages": [{"role": "user", "content": "Hello, Claude"}]}


def anthropic_sync():
    message = client.messages.create(**client_kwargs)
    print(message.content[0].text)


def anthropic_stream():
    message = client.messages.create(**client_kwargs, stream=True)
    for event in message:
        if isinstance(event, MessageStartEvent):
            print(f"{event.type}: {event.message.usage.input_tokens}")
        elif isinstance(event, ContentBlockDeltaEvent):
            print(f"{event.type}: {event.delta.text}")
        elif isinstance(event, MessageDeltaEvent):
            print(f"{event.type}: {event.usage.output_tokens}")
        else:
            print(f"{event.type}: {event}")


def anthropic_stream_context_manager():
    with client.messages.stream(**client_kwargs) as stream:
        for text in stream.text_stream:
            print(text, end="", flush=True)
        print()
        message = stream.get_final_message()
        print(message.model_dump_json(indent=2))


async def async_anthropic():
    message = await aclient.messages.create(**client_kwargs)
    print(message.content[0].text)


async def async_anthropic_stream():
    message = await aclient.messages.create(**client_kwargs, stream=True)
    async for event in message:
        if isinstance(event, MessageStartEvent):
            print(f"{event.type}: {event.message.usage.input_tokens}")
        elif isinstance(event, ContentBlockDeltaEvent):
            print(f"{event.type}: {event.delta.text}")
        elif isinstance(event, MessageDeltaEvent):
            print(f"{event.type}: {event.usage.output_tokens}")
        else:
            print(f"{event.type}: {event}")


async def async_anthropic_stream_context_manager():
    async with aclient.messages.stream(**client_kwargs) as stream:
        async for text in stream.text_stream:
            print(text, end="", flush=True)
        print()
        message = await stream.get_final_message()
        print(message.model_dump_json(indent=2))


if __name__ == "__main__":
    anthropic_sync()
    anthropic_stream()
    anthropic_stream_context_manager()
    asyncio.run(async_anthropic())
    asyncio.run(async_anthropic_stream())
    asyncio.run(async_anthropic_stream_context_manager())
