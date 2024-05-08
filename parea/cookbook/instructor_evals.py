import os

from dotenv import load_dotenv

from pydantic import BaseModel, field_validator
from openai import OpenAI
import instructor
from parea import Parea


load_dotenv()

client = OpenAI()

p = Parea(api_key=os.getenv("PAREA_API_KEY"))
p.wrap_openai_client(client, "instructor")

client = instructor.from_openai(client)


class User(BaseModel):
    name: str
    age: int

    @field_validator("name")
    def name_is_uppercase(cls, v: str):
        assert v.isupper(), "Name must be uppercase"
        return v


resp = client.messages.create(
    model="gpt-3.5-turbo",
    max_tokens=1024,
    max_retries=3,
    messages=[
        {
            "role": "user",
            "content": "Extract Bobby is 18 years old.",
        }
    ],
    response_model=User,
)

assert isinstance(resp, User)
assert resp.name == "BOBBY"  # due to validation
assert resp.age == 18
print(resp)
