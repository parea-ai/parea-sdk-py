from typing import Callable, List, Optional

from parea.evals.utils import call_openai, get_context, safe_json_loads
from parea.schemas.log import Log


def context_ranking_pointwise_factory(
    question_field: str = "question",
    context_fields: Optional[List[str]] = None,
    ranking_measurement="average_precision",
    model: Optional[str] = "gpt-3.5-turbo-16k",
    is_azure: Optional[bool] = False,
) -> Callable[[Log], float]:
    """
    This factory creates an evaluation function that measures how well the retrieved contexts are ranked by relevancy to the given query
    by pointwise estimation of the relevancy of every context to the query. It is based on the paper
    [RAGAS: Automated Evaluation of Retrieval Augmented Generation](https://arxiv.org/abs/2309.15217) which suggests using an LLM
    to check if every extracted context is relevant. Then, they measure how well the contexts are ranked by calculating the
    mean average precision. Note that this approach considers any two relevant contexts equally important/relevant to the query.

    Args:
        is_azure: Whether to use the Azure API. Defaults to False.
        model: The model which should be used for grading. Defaults to "gpt-3.5-turbo-16k".
        question_field: The key name/field used for the question/query of the user. Defaults to "question".
        context_fields: An optional list of key names/fields used for the retrieved contexts in the input to function. If empty list or None, it will use the output field of the log as context. Defaults to None.
        ranking_measurement: Method to calculate ranking. Currently, only supports "average_precision".

    Returns:
        Callable[[Log], float]: A function that takes a log as input and returns a score between 0 and 1 indicating
        how well the retrieved context is ranked by their relevancy.

    Raises:
        ImportError: If numpy is not installed.
    """
    try:
        import numpy as np
    except ImportError:
        raise ImportError("Please install numpy to use this metric.")

    def context_ranking_pointwise(log: Log) -> float:
        """Quantifies if the retrieved context is ranked by their relevancy"""
        question = log.inputs[question_field]
        contexts = get_context(log, context_fields, True)

        verifications = []
        for context in contexts:
            response = call_openai(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": f"""\
Verify if the information in the given context is useful in answering the question.

question: What are the health benefits of green tea?
context:
This article explores the rich history of tea cultivation in China, tracing its roots back to the ancient dynasties. It discusses how different regions have developed their unique tea varieties and brewing techniques. The article also delves into the cultural significance of tea in Chinese society and how it has become a symbol of hospitality and relaxation.
verification:
{{"reason":"The context, while informative about the history and cultural significance of tea in China, does not provide specific information about the health benefits of green tea. Thus, it is not useful for answering the question about health benefits.", "verdict":"No"}}

question: How does photosynthesis work in plants?
context:
Photosynthesis in plants is a complex process involving multiple steps. This paper details how chlorophyll within the chloroplasts absorbs sunlight, which then drives the chemical reaction converting carbon dioxide and water into glucose and oxygen. It explains the role of light and dark reactions and how ATP and NADPH are produced during these processes.
verification:
{{"reason":"This context is extremely relevant and useful for answering the question. It directly addresses the mechanisms of photosynthesis, explaining the key components and processes involved.", "verdict":"Yes"}}

question:{question}
context:
{context}
verification:""",
                    }
                ],
                temperature=0.0,
                is_azure=is_azure,
            )
            verifications.append(response)

        if ranking_measurement == "average_precision":
            response = [safe_json_loads(item) for item in verifications]
            response = [int("yes" in resp.get("verdict", " ").lower()) if resp.get("verdict") else np.nan for resp in response]
            denominator = sum(response) + 1e-10
            numerator = sum([(sum(response[: i + 1]) / (i + 1)) * response[i] for i in range(len(response))])
            return numerator / denominator
        else:
            raise NotImplementedError

    return context_ranking_pointwise
