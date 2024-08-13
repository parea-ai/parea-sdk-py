import os

import anthropic
import instructor
from dotenv import load_dotenv
from openai import AsyncOpenAI
from pydantic import BaseModel

from parea import Parea, trace

load_dotenv()

oai_aclient = AsyncOpenAI()
ant_client = anthropic.AsyncClient()

p = Parea(api_key=os.getenv("PAREA_API_KEY"))

p.wrap_openai_client(oai_aclient, "instructor")
p.wrap_anthropic_client(ant_client)

oai_aclient = instructor.from_openai(oai_aclient)
ant_client = instructor.from_anthropic(ant_client)


class UserDetail(BaseModel):
    name: str
    age: str


@trace
async def ainner_main():
    user = oai_aclient.completions.create_partial(
        model="gpt-4o-mini",
        max_tokens=1024,
        max_retries=3,
        messages=[
            {
                "role": "user",
                "content": "Please create a user",
            }
        ],
        response_model=UserDetail,
    )
    return user


async def amain():
    resp = await ainner_main()
    async for u in resp:
        print(u)


@trace
def inner_main():
    user = ant_client.completions.create_partial(
        model="claude-3-5-sonnet-20240620",
        max_tokens=1024,
        max_retries=3,
        messages=[
            {
                "role": "user",
                "content": "Please create a user",
            }
        ],
        response_model=UserDetail,
    )
    return user


def main():
    resp = inner_main()
    for u in resp:
        print(u)


if __name__ == "__main__":
    import asyncio

    asyncio.run(amain())

    main()
