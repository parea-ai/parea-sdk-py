import json
import os

from dotenv import load_dotenv

from parea import Parea, trace
from parea.evals import call_openai
from parea.schemas import Completion
from parea.schemas.log import EvaluationResult, Log

load_dotenv()

p = Parea(api_key=os.getenv("PAREA_API_KEY"))


def eval_fun(log: Log) -> EvaluationResult:
    # access the output and target from the log
    # output, target = log.output, log.target
    response: str = call_openai(
        model="gpt-4o",
        messages=[{"role": "system", "content": "Use JSON. provide a score and reason."}],  # <- CHANGE THIS
        response_format={"type": "json_object"},
        temperature=0.0,
    )
    response_dict = json.loads(response)
    return EvaluationResult(name="YOUR_EVAL_NAME", score=response_dict["score"], reason=response_dict["reason"])


@trace(eval_funcs=[eval_fun])
def deployed_prompt(prompt_template_input: str) -> str:
    return p.completion(Completion(deployment_id="YOUR_DEPLOYED_PROMPT_ID", llm_inputs={"prompt_template_input_name": prompt_template_input})).content


if __name__ == "__main__":
    p.experiment(
        name="some_experiment_name",
        data=172,  # dataset Id from Parea, can also use dataset name if unique
        func=deployed_prompt,
    ).run()
