from typing import Callable, Optional

import re

from parea.evals.utils import call_openai
from parea.schemas.log import Log


def likert_scale_factory(article_field: Optional[str] = "article", model: Optional[str] = "gpt-4", is_azure: Optional[bool] = False) -> Callable[[Log], float]:
    """
    This factory creates an evaluation function that grades the quality of a summary on a Likert scale from 1-5 along
    the dimensions of relevance, consistency, fluency, and coherence. It is based on the paper
    [Human-like Summarization Evaluation with ChatGPT](https://arxiv.org/abs/2304.02554) which finds that using `gpt-3.5-0301`
    leads to a highest correlation with human expert judgment when grading summaries on a Likert scale from 1-5 than baseline
    methods. Noteworthy is that [BARTScore](https://arxiv.org/abs/2106.11520) was very competitive to `gpt-3.5-0301`.

    Args:
        is_azure: Whether to use the Azure API. Defaults to False.
        article_field: The key name/field used for the content which should be summarized. Defaults to "article".
        model: The model which should be used for grading. Currently, only supports OpenAI chat models. Defaults to "gpt-4".

    Returns:
        Callable[[Log], float]: A function that takes a log as input and returns a score between 0 and 1 indicating
        the quality of the summary on a Likert scale from 1-5 along the dimensions of relevance, consistency, fluency, and coherence.
    """

    def likert_scale(log: Log) -> float:
        article = log.inputs[article_field]
        output = log.output
        prompt = f"""Evaluate the quality of summaries written for a news article. Rate each summary on four dimensions: relevance, consistency, fluency, and coherence. You should rate on a scale from 1 (worst) to 5 (best).

Definitions are as follows:
Relevance: The rating measures how well the summary captures the key points of the article. Consider whether all and only the important aspects are contained in the summary.
Consistency: The rating measures whether the facts in the summary are consistent with the facts in the original article. Consider whether the summary does reproduce all facts accurately and does not make up untrue information.
Fluency: This rating measures the quality of individual sentences, whether they are well-written and grammatically correct. Consider the quality of individual sentences.
Coherence: The rating measures the quality of all sentences collectively, to fit together and sound natural. Consider the quality of the summary as a whole.

The article and the summary are given below:
Article: {article}
Summary: {output}"""
        response = call_openai(
            model=model,
            messages=[
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            is_azure=is_azure,
        )

        # extract the scores
        pattern = re.compile(r"\d+")
        matches = pattern.findall(response)
        if matches:
            scores = matches
        else:
            scores = [0, 0, 0, 0]

        # normalize the scores
        scores = [float(score) / 5.0 for score in scores]

        # average the scores
        return sum(scores) / len(scores)

    return likert_scale
