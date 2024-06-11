import os

from dotenv import load_dotenv

from parea import Parea, trace
from parea.schemas import Completion

load_dotenv()

p = Parea(api_key=os.getenv("PAREA_API_KEY"))


@trace(eval_funcs_names=["YOUR_EVAL_NAME"])
def deployed_prompt(prompt_template_input: str) -> str:
    return p.completion(Completion(deployment_id="YOUR_DEPLOYED_PROMPT_ID", llm_inputs={"prompt_template_input_name": prompt_template_input})).content


if __name__ == "__main__":
    p.experiment(
        name="some_experiment_name",
        data=172,  # dataset Id from Parea, can also use dataset name if unique
        func=deployed_prompt,
    ).run()
