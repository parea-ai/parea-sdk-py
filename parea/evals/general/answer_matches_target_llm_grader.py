from typing import Callable, Optional

from parea.evals.utils import call_openai
from parea.schemas.log import Log


def answer_matches_target_llm_grader_factory(
    question_field: Optional[str] = "question",
    model: Optional[str] = "gpt-4",
) -> Callable[[Log], float]:
    """Quantifies how much the generated answer matches the ground truth / target."""

    def answer_matches_target_llm_grader(log: Log) -> float:
        question = log.inputs[question_field]
        output = log.output
        target = log.target
        response = call_openai(
            model=model,
            messages=[
                {"role": "system", "content": "You are CompareGPT, a machine to verify the groundedness of predictions. Answer with " "only yes/no."},
                {
                    "role": "user",
                    "content": f"""You are given a question, the corresponding ground-truth answer and a prediction from a model. Compare the "Ground-truth answer" and the "Prediction" to determine whether the prediction correctly answers the question. All information in the ground-truth answer must be present in the prediction, including numbers and dates. You must answer "no" if there are any specific details in the ground-truth answer that are not mentioned in the prediction. There should be no contradicting statements in the prediction. The prediction may contain extra information. If the prediction states something as a possibility, treat it as a definitive answer.

Question: {question}
Ground-truth answer: {target}
Prediction: {output}

CompareGPT response:""",
                },
            ],
            temperature=0.0,
        )
        return float("yes" in response.lower())

    return answer_matches_target_llm_grader
