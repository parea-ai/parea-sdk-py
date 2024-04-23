from typing import Callable, List, Optional

from parea.evals.utils import call_openai, get_context, sent_tokenize
from parea.schemas.log import Log


def context_query_relevancy_factory(
    question_field: str = "question", context_fields: Optional[List[str]] = None, model: Optional[str] = "gpt-3.5-turbo-16k", is_azure: Optional[bool] = False
) -> Callable[[Log], float]:
    """
    This factory creates an evaluation function that measures how relevant the retrieved context is to the given question.
    It is based on the paper [RAGAS: Automated Evaluation of Retrieval Augmented Generation](https://arxiv.org/abs/2309.15217)
    which suggests using an LLM to extract any sentence from the retrieved context relevant to the query. Then, calculate
    the ratio of relevant sentences to the total number of sentences in the retrieved context.

    Args:
        is_azure: Whether to use the Azure API. Defaults to False.
        model: The model which should be used for grading. Defaults to "gpt-3.5-turbo-16k".
        question_field: The key name/field used for the question/query of the user. Defaults to "question".
        context_fields: An optional list of key names/fields used for the retrieved contexts in the input to function. If empty list or None, it will use the output field of the log as context. Defaults to None.

    Returns:
        Callable[[Log], float]: A function that takes a log as input and returns a score between 0 and 1 indicating
        if the retrieved context is relevant to the query.
    """

    def context_query_relevancy(log: Log) -> float:
        """Quantifies how much the retrieved context relates to the query."""
        question = log.inputs[question_field]
        context = get_context(log, context_fields)

        extracted_sentences = call_openai(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": f"""\
Please extract relevant sentences from the provided context that is absolutely required answer the following question. If no relevant sentences are found, or if you believe the question cannot be answered from the given context, return the phrase "Insufficient Information".  While extracting candidate sentences you're not allowed to make any changes to sentences from given context.

question:{question}
context:\n{context}
candidate sentences:\n""",
                }
            ],
            temperature=0.0,
            is_azure=is_azure,
        ).strip()
        if "insufficient information" in extracted_sentences.lower() and abs(len(extracted_sentences) - len("insufficient information")) < 10:
            return 0.0
        else:
            n_extracted_sentences = len(sent_tokenize(extracted_sentences))
            n_context_sentences = len(sent_tokenize(context))
            return n_extracted_sentences / n_context_sentences

    return context_query_relevancy
