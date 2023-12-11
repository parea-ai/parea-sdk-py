from typing import Callable

import ast
import re

from parea.evals.utils import call_openai
from parea.schemas.log import Log

one_score_pattern = re.compile("\[\[(\d+\.?\d*)\]\]")
one_score_pattern_backup = re.compile("\[(\d+\.?\d*)\]")


def llm_grader_factory(model: str, question_field: str = "question") -> Callable[[Log], float]:
    """Measures the generated response quality by using a LLM on a scale of 1 to 10."""

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
