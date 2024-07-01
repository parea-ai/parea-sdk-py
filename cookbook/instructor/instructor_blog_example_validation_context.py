from typing import Annotated

import os
import re

import instructor
import requests
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import AfterValidator, BaseModel, ValidationInfo

from parea import Parea

load_dotenv()

client = OpenAI()

p = Parea(api_key=os.getenv("PAREA_API_KEY"))
p.wrap_openai_client(client, "instructor")

client = instructor.from_openai(client)


def check_urls(v, info: ValidationInfo):
    urls = re.findall(r"https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+", v)
    domain = info.context.get("domain") if info and info.context else None
    errors = []
    for url in urls:
        if domain and not url.startswith(domain):
            errors.append(f"URL {url} is not from useinstructor.com, Only include URLs that include use instructor.com. ")
        response = requests.get(url)
        if response.status_code != 200:
            errors.append(f"URL {url} returned status code {response.status_code}. Only include valid URLs that exist.")
        elif "404" in response.text:
            errors.append(f"URL {url} contained '404' in the body. Only include valid URLs that exist.")
    if errors:
        raise ValueError("\n".join(errors))
    return v


Body = Annotated[str, AfterValidator(check_urls)]


class Email(BaseModel):
    subject: str
    body: Body


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
        validation_context={"domain": "https://python.useinstructor.com"},
    )
    print(email)


if __name__ == "__main__":
    main()
