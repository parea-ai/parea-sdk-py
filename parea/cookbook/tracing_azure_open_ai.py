import asyncio
import os

from dotenv import load_dotenv
from openai.lib.azure import AsyncAzureOpenAI, AzureOpenAI

from parea import Parea, trace
from parea.cookbook.data.openai_input_examples import functions_example, simple_example

load_dotenv()

client = AzureOpenAI(
    api_version="2023-12-01-preview",
    api_key=os.getenv("AZURE_OAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OAI_ENDPOINT"),
)
aclient = AsyncAzureOpenAI(
    api_version="2023-12-01-preview",
    api_key=os.getenv("AZURE_OAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OAI_ENDPOINT"),
)

p = Parea(api_key=os.getenv("PAREA_API_KEY"))
p.wrap_openai_client(client)
p.wrap_openai_client(aclient)


@trace
def call_azure(data: dict):
    response = client.chat.completions.create(**data)
    print(response)


@trace
def call_azure_stream(data: dict):
    data["stream"] = True
    stream = client.chat.completions.create(**data)
    for chunk in stream:
        if chunk.choices:
            print(chunk.choices[0].delta or "")


@trace
async def acall_azure(data: dict):
    response = await aclient.chat.completions.create(**data)
    print(response)


@trace
async def acall_azure_stream(data: dict):
    data["stream"] = True
    stream = await aclient.chat.completions.create(**data)
    async for chunk in stream:
        if chunk.choices:
            print(chunk.choices[0].delta or "")


if __name__ == "__main__":
    azure_model = "AZURE_MODEL_NAME"  # replace with your model name
    functions_example["model"] = azure_model
    simple_example["model"] = azure_model
    call_azure(functions_example)
    # call_azure_stream(simple_example)
    # call_azure_stream(functions_example)
    asyncio.run(acall_azure(simple_example))
    # asyncio.run(acall_azure(functions_example))
    # asyncio.run(acall_azure_stream(simple_example))
    asyncio.run(acall_azure_stream(functions_example))
