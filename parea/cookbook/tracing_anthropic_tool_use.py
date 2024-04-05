import asyncio
import os

import anthropic
from dotenv import load_dotenv

from parea import Parea

load_dotenv()

client = anthropic.Anthropic()
aclient = anthropic.AsyncAnthropic()

p = Parea(api_key=os.getenv("PAREA_API_KEY"), project_name="testing")
p.wrap_anthropic_client(client)
p.wrap_anthropic_client(aclient)

client_kwargs = {
    "model": "claude-3-opus-20240229",
    "max_tokens": 1024,
    "messages": [{"role": "user", "content": "What's the weather like in San Francisco?"}],
    "tools": [
        {
            "name": "get_weather",
            "description": "Get the current weather in a given location",
            "input_schema": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    }
                },
                "required": ["location"],
            },
        }
    ]
}


def anthropic_sync():
    message = client.beta.tools.messages.create(**client_kwargs)

    print(message.content[0])


async def async_anthropic():
    message = await aclient.beta.tools.messages.create(**client_kwargs)
    print(message.content[0].text)



if __name__ == "__main__":
    # anthropic_sync()
    asyncio.run(async_anthropic())
