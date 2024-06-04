import os

from dotenv import load_dotenv

from parea import Parea, trace
from parea.schemas import Completion, LLMInputs, Log, Message, ModelParams, Role

load_dotenv()

p = Parea(api_key=os.getenv("PAREA_API_KEY"))


def eval_func(log: Log) -> float:
    from random import random
    from time import sleep

    sleep(random() * 10)
    return random()


# annotate the function with the trace decorator and pass the evaluation function(s)
@trace(eval_funcs=[eval_func])
def func(lang: str, framework: str) -> str:
    return p.completion(
        data=Completion(
            llm_configuration=LLMInputs(
                model="gpt-3.5-turbo",
                model_params=ModelParams(temp=1),
                messages=[
                    Message(role=Role.user, content=f"Write a hello world program in {lang} using {framework}"),
                ],
            )
        )
    ).content


if __name__ == "__main__":
    p.experiment(
        data="Hello World Example",  # this is the name of your Dataset in Parea (Dataset page)
        func=func,
    ).run(name="hello-world-example")

    # Or use a dataset using its ID instead of the name
    # p.experiment(
    #     data=121,  # this is the id of your Dataset in Parea (Dataset page)
    #     func=func,
    # ).run(name="hello-world-example")
