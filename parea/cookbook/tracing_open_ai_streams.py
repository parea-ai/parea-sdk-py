import asyncio
import os

from dotenv import load_dotenv
from openai import AsyncOpenAI, OpenAI

from parea import Parea, trace
from parea.cookbook.data.openai_input_examples import functions_example, simple_example_json

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
aclient = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

p = Parea(api_key=os.getenv("PAREA_API_KEY"))
p.wrap_openai_client(client)
p.wrap_openai_client(aclient)


@trace
def call_openai_stream(data: dict):
    data["stream"] = True
    stream = client.chat.completions.create(**data)
    for chunk in stream:
        print(chunk.choices[0].delta or "")


@trace
async def acall_openai_stream(data: dict):
    data["stream"] = True
    stream = await aclient.chat.completions.create(**data)
    async for chunk in stream:
        print(chunk.choices[0].delta or "")


if __name__ == "__main__":
    # call_openai_stream(simple_example)
    call_openai_stream(simple_example_json)
    # call_openai_stream(functions_example)
    # asyncio.run(acall_openai_stream(simple_example))
    asyncio.run(acall_openai_stream(functions_example))
