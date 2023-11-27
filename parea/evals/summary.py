from typing import Callable, Optional

import re

from parea.evals.utils import call_openai
from parea.schemas.models import Log


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


def factual_inconsistency_scale_factory(
    article_field: Optional[str] = "article",
    model: Optional[str] = "gpt-4",
) -> Callable[[Log], float]:
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
        )

        pattern = re.compile(r"\d+")
        match = pattern.search(response)
        if match:
            score = match.group()
        else:
            score = 0

        return float(score) / 10.0

    return factual_inconsistency_scale


def likert_scale_eval_factory(
    article_field: Optional[str] = "article",
    model: Optional[str] = "gpt-4",
) -> Callable[[Log], float]:
    def likert_scale_eval(log: Log) -> float:
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

    return likert_scale_eval
