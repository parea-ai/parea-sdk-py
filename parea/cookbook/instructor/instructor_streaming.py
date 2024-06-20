import os

import instructor
from dotenv import load_dotenv
from openai import AsyncOpenAI


from parea import Parea

load_dotenv()

client = AsyncOpenAI()

p = Parea(api_key=os.getenv("PAREA_API_KEY"))
p.wrap_openai_client(client, "instructor")

client = instructor.from_openai(client)


from pydantic import BaseModel



class UserDetail(BaseModel):
    name: str
    age: int


async def main():
    user = client.completions.create_partial(
        model="gpt-3.5-turbo",
        max_tokens=1024,
        max_retries=3,
        messages=[
            {
                "role": "user",
                "content": "Please crea a user",
            }
        ],
        response_model=UserDetail,
    )
    # print(user)
    async for u in user:
        print(u)

    user2 = client.completions.create_partial(
        model="gpt-3.5-turbo",
        max_tokens=1024,
        max_retries=3,
        messages=[
            {
                "role": "user",
                "content": "Please crea a user",
            }
        ],
        response_model=UserDetail,
    )
    async for u in user2:
        print(u)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
