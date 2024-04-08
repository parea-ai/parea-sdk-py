from typing import Callable, Optional

import json

from parea.evals import call_openai
from parea.schemas import Log


def context_has_answer_factory(question_field: Optional[str] = "question", model: Optional[str] = "gpt-3.5-turbo-0125", is_azure: Optional[bool] = False) -> Callable[[Log], bool]:
    """
    This factory creates an evaluation metric which assess whether the given context has the answer to the given question.
    It is useful to measure the performance of a model in a question-answering task by measuring Hit Rate without the need to know the correct answer.

    Args:
        question_field: The key name/field used for the question/query of the user. Defaults to "question".
        model: The model which should be used for grading. Currently, only supports OpenAI chat models. Defaults to "gpt-3.5-turbo-0125".
        is_azure: Whether to use the Azure API. Defaults to False.

    Returns:
        Callable[[Log], bool]: A function that takes a log as input and returns a boolean indicating if the context has the answer to the given question.
    """

    def context_has_answer(log: Log) -> bool:
        question = log.inputs[question_field]
        answer = str(log.output)

        formatted_messages = [
            {
                "role": "user",
                "content": f"""You are given a question and a list of answers. The answers were retrieved from a database which contains the question answer pairs. You need to decide if any of the given answers is the answer to the given question.

Question:
{question}

Answers:
{answer}

Answer in the following JSON format:
{{"thoughts": "<thoughts>", "final_verdict": "<true|false>"}}""",
            }
        ]

        response = call_openai(model=model, temperature=0.0, messages=formatted_messages, response_format={"type": "json_object"}, is_azure=is_azure)
        final_verdict = json.loads(response).get("final_verdict", "").lower()
        return final_verdict == "true"

    return context_has_answer
