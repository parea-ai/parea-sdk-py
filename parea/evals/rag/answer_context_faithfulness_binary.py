from typing import Callable, Optional

from parea.evals.utils import call_openai
from parea.schemas.log import Log


def answer_context_faithfulness_binary_factory(
    question_field: Optional[str] = "question",
    context_field: Optional[str] = "context",
    model: Optional[str] = "gpt-4",
) -> Callable[[Log], float]:
    """Quantifies how much the generated answer can be inferred from the retrieved context."""

    def answer_context_faithfulness_binary(log: Log) -> float:
        question = log.inputs[question_field]
        evidence = log.inputs[context_field]
        output = log.output
        response = call_openai(
            model=model,
            messages=[
                {"role": "system", "content": "You are CompareGPT, a machine to verify the groundedness of predictions. Answer with " "only yes/no."},
                {
                    "role": "user",
                    "content": f"You are given a question, the corresponding evidence and a prediction from a model. Compare "
                    f'the "Prediction" and the "Evidence" to determine whether all the information of the '
                    f"prediction in present in the evidence or can be inferred from the evidence. You must answer "
                    f'"no" if there are any specific details in the prediction that are not mentioned in the '
                    f"evidence or cannot be inferred from the evidence.\n\n"
                    f"Question: {question}\n\nPrediction: {output}\n\nEvidence: {evidence}\n\nCompareGPT response:",
                },
            ],
            temperature=0.0,
        )
        return float("yes" in response.lower())

    return answer_context_faithfulness_binary
