import os

from dotenv import load_dotenv

from parea import Parea, trace
from parea.schemas.log import EvaluationResult, Log

load_dotenv()

p = Parea(api_key=os.getenv("PAREA_API_KEY"))


def eval_func_with_reason(log: Log) -> EvaluationResult:
    if log.output == log.target:
        return EvaluationResult(name="matches_target", score=1.0, reason="Output matches target")
    elif "Hello" in log.target and "Hello" not in log.output:
        return EvaluationResult(name="matches_target", score=0, reason="Output misses 'Hello'")
    elif "Hi" in log.target and "Hi" not in log.output:
        return EvaluationResult(name="matches_target", score=0, reason="Output misses 'Hi'")
    else:
        return EvaluationResult(name="matches_target", score=0, reason="Output does not match target")


# annotate the function with the trace decorator and pass the evaluation function(s)
@trace(eval_funcs=[eval_func_with_reason])
def greeting(name: str) -> str:
    return f"Hello {name}"


data = [
    {
        "name": "Foo",
        "target": "Hi Foo",
    },
    {
        "name": "Bar",
        "target": "Hello Bar",
    },
]  # test data to run the experiment on (list of dicts)


# You can optionally run the experiment manually by calling `.run()`
if __name__ == "__main__":
    p.experiment(
        name="greeting",
        data=data,
        func=greeting,
        n_trials=1,
    ).run()
