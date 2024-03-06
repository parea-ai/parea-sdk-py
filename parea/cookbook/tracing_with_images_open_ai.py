from typing import Optional

import json
import os

from dotenv import load_dotenv
from openai import OpenAI

from parea import Parea, trace, trace_insert
from parea.schemas import TraceLogImage

load_dotenv()


client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

p = Parea(api_key=os.getenv("PAREA_API_KEY"))
p.wrap_openai_client(client)


@trace
def image_maker(query: str) -> str:
    response = client.images.generate(prompt=query, model="dall-e-3")
    image_url = response.data[0].url
    caption = {"original_prompt": query, "revised_prompt": response.data[0].revised_prompt}
    trace_insert({"images": [TraceLogImage(url=image_url, caption=json.dumps(caption))]})
    return image_url


@trace
def ask_vision(image_url: str) -> Optional[str]:
    response = client.chat.completions.create(
        model="gpt-4-vision-preview",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What’s in this image?"},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            }
        ],
        max_tokens=300,
    )
    return response.choices[0].message.content


@trace
def main(query: str) -> str:
    image_url = image_maker(query)
    return ask_vision(image_url)


if __name__ == "__main__":
    result = main("A cat sitting comfortably on a chair")
    print(result)
