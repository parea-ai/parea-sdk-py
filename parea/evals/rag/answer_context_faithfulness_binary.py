from typing import Callable, Optional

from parea.evals.utils import call_openai
from parea.schemas.log import Log


def answer_context_faithfulness_binary_factory(
    question_field: Optional[str] = "question", context_field: Optional[str] = "context", model: Optional[str] = "gpt-3.5-turbo-16k", is_azure: Optional[bool] = False
) -> Callable[[Log], float]:
    """
    This factory creates an evaluation function that classifies if the generated answer was faithful to the given context.
    It is based on the paper [Evaluating Correctness and Faithfulness of Instruction-Following Models for Question Answering](https://arxiv.org/abs/2307.16877)
    which suggests using an LLM to flag any information in the generated answer that cannot be deduced from the given context.
    They find that GPT-4 is the best model for this analysis as measured by correlation with human judgment.

    Args:
        is_azure: Whether to use the Azure API. Defaults to False.
        question_field: The key name/field used for the question/query of the user. Defaults to "question".
        context_field: The key name/field used for the retrieved context. Defaults to "context".
        model: The model which should be used for grading. Currently, only supports OpenAI chat models. Defaults to "gpt-4".

    Returns:
        Callable[[Log], float]: A function that takes a log as input and returns a score between 0 and 1 indicating
        if the generated answer was faithful to the given context.
    """

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
            is_azure=is_azure,
        )
        return float("yes" in response.lower())

    return answer_context_faithfulness_binary
