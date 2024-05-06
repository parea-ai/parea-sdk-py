import asyncio
import os
import uuid

from dotenv import load_dotenv

from parea import Parea, trace
from parea.schemas import Completion, LLMInputs, Log, Message, ModelParams, Role

load_dotenv()

p = Parea(api_key=os.getenv("PAREA_API_KEY"))


DATA = [{"topic": "Python"}, {"topic": "Javascript"}, {"topic": "Water"}, {"topic": "Fire"}]
models = ["gpt-4-turbo", "claude-3-haiku-20240307"]


def eval_func(log: Log) -> float:
    from random import random

    return random()


def model_call_factory(model: str):
    @trace(eval_funcs=[eval_func])
    def func(topic: str) -> str:
        return p.completion(
            data=Completion(
                llm_configuration=LLMInputs(
                    model=model,
                    model_params=ModelParams(temp=1),
                    messages=[Message(role=Role.user, content=f"Write a short haiku about {topic}")],
                )
            )
        ).content

    return func


async def main():
    await asyncio.gather(
        *[p.experiment(name="Write-Haikus", data=DATA, func=model_call_factory(model), n_trials=4).arun(run_name=f"{model}-{str(uuid.uuid4())[:4]}") for model in models]
    )


if __name__ == "__main__":
    asyncio.run(main())
