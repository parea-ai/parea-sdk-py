import os
import re

import instructor
import requests
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field, field_validator

from parea import Parea

load_dotenv()

client = OpenAI()

p = Parea(api_key=os.getenv("PAREA_API_KEY"))
p.wrap_openai_client(client, "instructor")

client = instructor.from_openai(client)


class Email(BaseModel):
    subject: str
    body: str = Field(
        ...,
        description="Email body, Should contain links to instructor documentation. ",
    )

    @field_validator("body")
    def check_urls(cls, v):
        urls = re.findall(r"https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+", v)
        errors = []
        for url in urls:
            if not url.startswith("https://python.useinstructor.com"):
                errors.append(f"URL {url} is not from useinstructor.com, Only include URLs that include use instructor.com. ")
            response = requests.get(url)
            if response.status_code != 200:
                errors.append(f"URL {url} returned status code {response.status_code}. Only include valid URLs that exist.")
            elif "404" in response.text:
                errors.append(f"URL {url} contained '404' in the body. Only include valid URLs that exist.")
        if errors:
            raise ValueError("\n".join(errors))
        return v


def main():
    email = client.messages.create(
        model="gpt-3.5-turbo",
        max_tokens=1024,
        max_retries=3,
        messages=[
            {
                "role": "user",
                "content": "I'm responding to a student's question. Here is the link to the documentation: {{doc_link1}} and {{doc_link2}}",
            }
        ],
        template_inputs={
            "doc_link1": "https://python.useinstructor.com/docs/tutorial/tutorial-1",
            "doc_link2": "https://jxnl.github.io/docs/tutorial/tutorial-2",
        },
        response_model=Email,
    )
    print(email)


if __name__ == "__main__":
    main()
