from typing import Callable, Optional

from parea.evals.utils import call_openai
from parea.schemas.log import Log


def factual_inconsistency_binary_factory(
    article_field: Optional[str] = "article",
    model: Optional[str] = "gpt-4",
) -> Callable[[Log], float]:
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
        )
        return float("yes" in response.lower())

    return factual_inconsistency_binary
