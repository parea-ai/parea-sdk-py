from typing import Callable, Optional

import ast
import re

from parea.evals.utils import call_openai
from parea.schemas.log import Log

one_score_pattern = re.compile(r"\[\[(\d+\.?\d*)\]\]")
one_score_pattern_backup = re.compile(r"\[(\d+\.?\d*)\]")


def llm_grader_factory(model: str = "gpt-4", question_field: str = "question", is_azure: Optional[bool] = False) -> Callable[[Log], float]:
    """
    This factory creates an evaluation function that uses an LLM to grade the response of an LLM to a given question.
    It is based on the paper [Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena](https://arxiv.org/abs/2306.05685)
    which introduces general-purpose zero-shot prompt to rate responses from an LLM to a given question on a scale from 1-10.
    They find that GPT-4's ratings agree as much with a human rater as a human annotator agrees with another one (>80%).
    Further, they observe that the agreement with a human annotator increases as the response rating gets clearer.
    Additionally, they investigated how much the evaluating LLM overestimated its responses and found that GPT-4 and
    Claude-1 were the only models that didn't overestimate themselves.

    Args:
        is_azure: Whether to use the Azure API. Defaults to False.
        model: The model which should be used for grading. Currently, only supports OpenAI chat models.
        question_field: The key name/field used for the question/query of the user. Defaults to "question".

    Returns:
        Callable[[Log], float]: A function that takes a log as input and returns a score between 0 and 1 which is the
        rating of the response on a scale from 1-10 divided by 10.
    """

    def llm_grader(log: Log) -> float:
        question = log.inputs[question_field]
        output = log.output
        rating_response = call_openai(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": f"[Instruction]\nPlease act as an impartial judge and evaluate the quality of the response "
                    f"provided by an AI assistant to the user question displayed below. Your evaluation should "
                    f"consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and "
                    f"level of detail of the response. Begin your evaluation by providing a short explanation. "
                    f"Be as objective as possible. After providing your explanation, you must rate the response "
                    f'on a scale of 1 to 10 by strictly following this format: "[[rating]]", for example: '
                    f'"Rating: [[5]]".\n\n[Question]\n{question}\n\n[The Start of Assistant\'s Answer]'
                    f"\n{output}\n[The End of Assistant's Answer]",
                },
            ],
            temperature=0.0,
            is_azure=is_azure,
        )
        match = re.search(one_score_pattern, rating_response)
        if not match:
            match = re.search(one_score_pattern_backup, rating_response)

        if match:
            rating = ast.literal_eval(match.groups()[0])
        else:
            rating = 0

        return rating / 10.0

    return llm_grader


llm_grader_gpt4 = llm_grader_factory("gpt-4")
llm_grader_gpt3t = llm_grader_factory("gpt-3.5-turbo-16k")
