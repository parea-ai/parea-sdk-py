import json
import os

from anthropic import Anthropic
from dotenv import load_dotenv
from openai import OpenAI

from parea import Parea, trace, trace_insert
from parea.schemas import TraceLogImage

load_dotenv()


oai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
a_client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

p = Parea(api_key=os.getenv("PAREA_API_KEY"))
p.wrap_openai_client(oai_client)
p.wrap_anthropic_client(a_client)


@trace
def image_maker(query: str) -> str:
    response = oai_client.images.generate(prompt=query, model="dall-e-3")
    image_url = response.data[0].url
    caption = {"original_prompt": query, "revised_prompt": response.data[0].revised_prompt}
    trace_insert({"images": [TraceLogImage(url=image_url, caption=json.dumps(caption))]})
    return image_url


from typing import Optional

import base64

import requests


@trace
def ask_vision(image_url: str) -> Optional[str]:
    image_data = requests.get(image_url).content
    base64_image = base64.b64encode(image_data).decode("utf-8")

    response = a_client.messages.create(
        model="claude-3-haiku-20240307",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": base64_image,
                        },
                    },
                    {"type": "text", "text": "What’s in this image?"},
                ],
            }
        ],
        max_tokens=300,
    )
    return response.content[0].text


@trace
def main(query: str) -> str:
    image_url = image_maker(query)
    return ask_vision(image_url)


if __name__ == "__main__":
    result = main("A dog sitting comfortably on a chair")
    print(result)
