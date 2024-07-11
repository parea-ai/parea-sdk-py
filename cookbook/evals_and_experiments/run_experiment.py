import os

from dotenv import load_dotenv

from parea import Parea, trace
from parea.schemas import Log

load_dotenv()

p = Parea(api_key=os.getenv("PAREA_API_KEY"))


def eval_func(log: Log) -> float:
    from random import random

    return random()


def eval_func2(log: Log) -> float:
    from random import random

    return random()


# annotate the function with the trace decorator and pass the evaluation function(s)
@trace(eval_funcs=[eval_func, eval_func2])
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


# # Define the experiment
# # You can use the CLI command "parea experiment parea/cookbook/run_experiment.py" to execute this experiment
# # or call `.run()`
# # p.experiment(
# #     data=data,  # Data to run the experiment on (list of dicts)
# #     func=greeting,  # Function to run (callable)
# #     n_trials=1,  # Number of times to run the experiment on the same data
# # )

# You can optionally run the experiment manually by calling `.run()`
if __name__ == "__main__":
    p.experiment(
        name="greeting",
        data=data,
        func=greeting,
    ).run()
