import asyncio
import os

import anthropic
from dotenv import load_dotenv

from parea import Parea
from parea.cookbook.data.anthropic_tool_use_examples import missing_information, multiple_tool_use, single_tool_use

load_dotenv()

client = anthropic.Anthropic()
aclient = anthropic.AsyncAnthropic()

p = Parea(api_key=os.getenv("PAREA_API_KEY"), project_name="testing")
p.wrap_anthropic_client(client)
p.wrap_anthropic_client(aclient)


def anthropic_sync(create_kwargs):
    message = client.beta.tools.messages.create(**create_kwargs)
    print(message.content)


async def async_anthropic(create_kwargs):
    message = await aclient.beta.tools.messages.create(**create_kwargs)
    print(message.content)


if __name__ == "__main__":
    anthropic_sync(single_tool_use)
    anthropic_sync(multiple_tool_use)
    anthropic_sync(missing_information)
    # asyncio.run(async_anthropic(single_tool_use))
    # asyncio.run(async_anthropic(multiple_tool_use))
    # asyncio.run(async_anthropic(missing_information))
