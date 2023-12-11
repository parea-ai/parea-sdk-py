from typing import Callable, Optional

import re

from parea.evals.utils import call_openai
from parea.schemas.log import Log


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
