from typing import Union

import json
import os

from dotenv import load_dotenv

from parea import Parea, trace
from parea.evals.general.levenshtein import levenshtein_distance
from parea.schemas import Log

load_dotenv()

p = Parea(api_key=os.getenv("PAREA_API_KEY"))


# evaluation function for the substep
def eval_choose_greeting(log: Log) -> Union[float, None]:
    if not (target := log.target):
        return None

    target_substep = json.loads(target)["substep"]  # log.target is a string
    output = log.output
    return levenshtein_distance(target_substep, output)


# sub-step
@trace(eval_funcs=[eval_choose_greeting])
def choose_greeting(name: str) -> str:
    return "Hello"


# end-to-end evaluation function
def eval_greet(log: Log) -> Union[float, None]:
    if not (target := log.target):
        return None

    target_overall = json.loads(target)["overall"]
    output = log.output
    return levenshtein_distance(target_overall, output)


@trace(eval_funcs=[eval_greet])
def greet(name: str) -> str:
    greeting = choose_greeting(name)
    return f"{greeting} {name}"


data = [
    {
        "name": "Foo",
        "target": {
            "overall": "Hi Foo",
            "substep": "Hi",
        },
    },
    {
        "name": "Bar",
        "target": {
            "overall": "Hello Bar",
            "substep": "Hello",
        },
    },
]


if __name__ == "__main__":
    p.experiment(
        name="greeting",
        data=data,
        func=greet,
    ).run()
