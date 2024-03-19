import asyncio
import os

from dotenv import load_dotenv

from parea import Parea, trace
from parea.schemas import Completion, LLMInputs, Message, ModelParams, Role

load_dotenv()

p = Parea(api_key=os.getenv("PAREA_API_KEY"))

completion = Completion(
    llm_configuration=LLMInputs(
        model="gpt-3.5-turbo-1106",
        model_params=ModelParams(temp=0.1),
        messages=[Message(role=Role.user, content="Write a short haiku about the moon.")],
    )
)


@trace
def call_llm_stream():
    stream = p.stream(completion)
    for chunk in stream:
        print(chunk)


@trace
async def acall_llm_stream():
    stream = p.astream(completion)
    async for chunk in stream:
        print(chunk)


if __name__ == "__main__":
    call_llm_stream()
    # asyncio.run(acall_llm_stream())
