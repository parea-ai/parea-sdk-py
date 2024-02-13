import os

from dotenv import load_dotenv

from parea import Parea, trace
from parea.evals import call_openai
from parea.schemas import Log

load_dotenv()

p = Parea(api_key=os.getenv("PAREA_API_KEY"))


# Evaluation function(s)
def is_between_1_and_n(log: Log) -> float:
    """Evaluates if the number is between 1 and n"""
    n = log.inputs["n"]
    try:
        return 1.0 if 1.0 <= float(log.output) <= float(n) else 0.0
    except ValueError:
        return 0.0


# annotate the function with the trace decorator and pass the evaluation function(s)
@trace(eval_funcs=[is_between_1_and_n])
def generate_random_number(n: str) -> str:
    return call_openai(
        [
            {"role": "user", "content": f"Generate a number between 1 and {n}."},
        ],
        model="gpt-3.5-turbo",
    )


# Define the experiment
# You can use the CLI command "parea experiment parea/cookbook/run_experiment.py" to execute this experiment
# or call `.run()`
# p.experiment(
#     data=[{"n": "11"}],  # Data to run the experiment on (list of dicts)
#     func=generate_random_number,  # Function to run (callable)
#     n_trials=1,  # Number of times to run the experiment on the same data
# )

# You can optionally run the experiment manually by calling `.run()`
if __name__ == "__main__":
    p.experiment(
        data=[{"n": "10"}],
        func=generate_random_number,
        n_trials=3,
    ).run()
