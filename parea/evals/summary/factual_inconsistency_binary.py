from typing import Callable, Optional

from parea.evals.utils import call_openai
from parea.schemas.log import Log


def factual_inconsistency_binary_factory(article_field: Optional[str] = "article", model: Optional[str] = "gpt-4", is_azure: Optional[bool] = False) -> Callable[[Log], float]:
    """
    This factory creates an evaluation function that classifies if a summary is factually inconsistent with the original text.
    It is based on the paper [ChatGPT as a Factual Inconsistency Evaluator for Text Summarization](https://arxiv.org/abs/2303.15621)
    which suggests using an LLM to assess the factuality of a summary by measuring how consistent the summary is with
    the original text, posed as a binary classification. They find that `gpt-3.5-turbo-0301` outperforms
    baseline methods such as SummaC and QuestEval when identifying factually inconsistent summaries.

    Args:
        article_field: The key name/field used for the content which should be summarized. Defaults to "article".
        model: The model which should be used for grading. Currently, only supports OpenAI chat models. Defaults to "gpt-4".
        is_azure: Whether to use the Azure API. Defaults to False.

    Returns:
        Callable[[Log], float]: A function that takes a log as input and returns a score between 0 and 1 indicating
        if the generated summary is factually consistent with the original text.
    """

    def factual_inconsistency_binary(log: Log) -> float:
        article = log.inputs[article_field]
        output = log.output
        prompt = f"""Decide if the following summary is consistent with the corresponding article. Note that consistency means all information in the summary is supported by the article.
    Article: {article}
    Summary: {output}
    Explain your reasoning step by step then answer (yes or no) the question:"""
        response = call_openai(
            model=model,
            messages=[
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            is_azure=is_azure,
        )
        return float("yes" in response.lower())

    return factual_inconsistency_binary
