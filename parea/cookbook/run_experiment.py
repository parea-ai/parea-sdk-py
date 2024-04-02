import os

from dotenv import load_dotenv

from parea import Parea, trace
from parea.evals.general import levenshtein

load_dotenv()

p = Parea(api_key=os.getenv("PAREA_API_KEY"))


# annotate the function with the trace decorator and pass the evaluation function(s)
@trace(eval_funcs=[levenshtein])
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
        n_trials=3,
    ).run()
