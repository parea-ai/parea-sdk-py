from typing import Callable, Optional

import re

from parea.evals.utils import call_openai
from parea.schemas.log import Log


def factual_inconsistency_scale_factory(article_field: Optional[str] = "article", model: Optional[str] = "gpt-4", is_azure: Optional[bool] = False) -> Callable[[Log], float]:
    """
    This factory creates an evaluation function that grades the factual consistency of a summary with the article on a scale from 1 to 10.
    It is based on the paper [ChatGPT as a Factual Inconsistency Evaluator for Text Summarization](https://arxiv.org/abs/2303.15621)
    which finds that using `gpt-3.5-turbo-0301` leads to a higher correlation with human expert judgment when grading
    the factuality of summaries on a scale from 1 to 10 than baseline methods such as SummaC and QuestEval.

    Args:
        article_field: The key name/field used for the content which should be summarized. Defaults to "article".
        model: The model which should be used for grading. Currently, only supports OpenAI chat models. Defaults to "gpt-4".
        is_azure: Whether to use the Azure API. Defaults to False.

    Returns:
        Callable[[Log], float]: A function that takes a log as input and returns a score between 0 and 1 indicating
        if the generated summary is factually consistent with the original text.
    """

    def factual_inconsistency_scale(log: Log) -> float:
        article = log.inputs[article_field]
        output = log.output
        prompt = f"""Score the following summary given the corresponding article with respect to consistency from 1 to 10. Note that consistency measures how much information included in the summary is present in the source article. 10 points indicate the summary contains only statements that are entailed by the source document.
    Article: {article}
    Summary: {output}
    Marks: """
        response = call_openai(
            model=model,
            messages=[
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            is_azure=is_azure,
        )

        pattern = re.compile(r"\d+")
        match = pattern.search(response)
        if match:
            score = match.group()
        else:
            score = 0

        return float(score) / 10.0

    return factual_inconsistency_scale
