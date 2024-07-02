from typing import List

import os

from dotenv import load_dotenv
from openai import OpenAI

from parea import Parea, trace

load_dotenv()

p = Parea(api_key=os.getenv("PAREA_API_KEY"))


class LLMCaller:
    def __init__(self, messages: List[dict[str, str]]):
        self.messages = messages
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        p.wrap_openai_client(self.client)

    @trace
    def __call__(self, model: str = "gpt-4o", temperature: float = 0.0) -> str:
        return self.client.chat.completions.create(model=model, temperature=temperature, messages=self.messages).choices[0].message.content


@trace
def main(topic: str) -> str:
    caller = LLMCaller(
        messages=[
            {"role": "system", "content": "You are a debater making an argument on a topic."},
            {"role": "user", "content": f"The discussion topic is {topic}"},
        ]
    )
    return caller()


if __name__ == "__main__":
    result = main("The impact of climate change on the economy")
    print(result)
