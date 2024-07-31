import random
from typing import List
import os

from dotenv import load_dotenv

from parea import Parea, trace
from parea.schemas import EvaluatedLog, Log, EvaluationResult

load_dotenv()

p = Parea(api_key=os.getenv("PAREA_API_KEY"))


def random_eval_factory(trial: int):
    def random_eval(log: Log) -> EvaluationResult:
        return EvaluationResult(
            score=1 if random.random() < 0.5 else 0,
            name=f'random_eval_{trial}'
        )
    return random_eval


# apply random evaluation function twice
@trace(eval_funcs=[random_eval_factory(1), random_eval_factory(2)])
async def starts_with_f(name: str) -> str:
    if name == "Foo":
        return "1"
    return "0"


# dataset-level evaluation function which checks if both random evaluations agree
def percent_evals_agree(logs: List[EvaluatedLog]) -> float:
    correct = 0
    total = 0
    for log in logs:
        if log.scores[0].score == log.scores[1].score:
            correct += 1
        total += 1
    return correct / total


data = [
    {
        "name": "Foo",
        "target": "1",
    },
    {
        "name": "Bar",
        "target": "0",
    },
    {
        "name": "Far",
        "target": "1",
    },
]  # test data to run the experiment on (list of dicts)


# You can optionally run the experiment manually by calling `.run()`
if __name__ == "__main__":
    p.experiment(
        name="Greeting",
        data=data,
        func=starts_with_f,
        dataset_level_evals=[percent_evals_agree]
    ).run()
