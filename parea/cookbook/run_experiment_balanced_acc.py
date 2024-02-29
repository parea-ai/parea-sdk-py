import asyncio
import os
from collections import defaultdict

from dotenv import load_dotenv

from parea import Parea, trace
from parea.schemas import EvaluatedLog, Log

load_dotenv()

p = Parea(api_key=os.getenv("PAREA_API_KEY"))


def is_correct(log: Log) -> bool:
    return log.target == log.output


def balanced_acc_is_correct(logs: list[EvaluatedLog]) -> float:
    score_name = is_correct.__name__

    correct = defaultdict(int)
    total = defaultdict(int)
    for log in logs:
        if (eval_result := log.get_score(score_name)) is not None:
            correct[log.target] += int(eval_result.score)
            total[log.target] += 1
    recalls = [correct[key] / total[key] for key in correct]

    return sum(recalls) / len(recalls)


# or use the pre-built `balanced_acc_factory` to create the function
# from parea.evals.dataset_level import balanced_acc_factory
#
#
# balanced_acc_is_correct = balanced_acc_factory(is_correct.__name__)


@trace(eval_funcs=[is_correct])
async def starts_with_f(name: str) -> str:
    await asyncio.sleep(1)
    if name == "Foo":
        return "1"
    return "0"


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
    p.experiment(data=data, func=starts_with_f, dataset_level_evals=[balanced_acc_is_correct], n_workers=2).run()
