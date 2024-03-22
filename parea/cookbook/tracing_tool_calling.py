import os

from dotenv import load_dotenv
from openai import OpenAI

from parea import Parea

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

p = Parea(api_key=os.getenv("PAREA_API_KEY"))
p.wrap_openai_client(client)


tools = [
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
]
messages = [{"role": "user", "content": "What's the weather like in Boston today?"}]
completion = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=messages,
    tools=tools,
    tool_choice="auto",
)
messages.append({k: v for k, v in completion.choices[0].message.model_dump().items() if v is not None})
# messages.append(completion.choices[0].message)
messages.append({"role": "tool", "content": "5 Celcius", "tool_call_id": completion.choices[0].message.tool_calls[0].id})
messages.append(
    {
        "role": "user",
        "content": "What's the weather like in Boston today?",
    }
)

final_completion = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=messages,
    tools=tools,
    tool_choice="auto",
)

print(final_completion)
