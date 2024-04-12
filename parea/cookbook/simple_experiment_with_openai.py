import os

from dotenv import load_dotenv
from openai import OpenAI

from parea import Parea, trace
from parea.schemas import Log

load_dotenv()

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
p = Parea(api_key=os.getenv("PAREA_API_KEY"))
p.wrap_openai_client(client)


def eval_func(log: Log) -> float:
    from random import random
    from time import sleep

    sleep(random() * 10)
    return random()


@trace(eval_funcs=[eval_func])
def func(lang: str, framework: str) -> str:
    return (
        client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {
                    "role": "user",
                    "content": f"Write a hello world program in {lang} using {framework}",
                }
            ],
        )
        .choices[0]
        .message.content
    )


if __name__ == "__main__":
    p.experiment(
        name="hello-world-example",
        data=[{"lang": "Python", "framework": "Flask"}],
        func=func,
    ).run()
