from typing import Callable, List

from parea.evals.utils import call_openai, sent_tokenize
from parea.schemas.log import Log


def context_query_relevancy_factory(question_field: str = "question", context_fields: list[str] = ["context"]) -> Callable[[Log], float]:
    """
    This factory creates an evaluation function that measures how relevant the retrieved context is to the given question.
    It is based on the paper [RAGAS: Automated Evaluation of Retrieval Augmented Generation](https://arxiv.org/abs/2309.15217)
    which suggests using an LLM to extract any sentence from the retrieved context relevant to the query. Then, calculate
    the ratio of relevant sentences to the total number of sentences in the retrieved context.

    Args:
        question_field: The key name/field used for the question/query of the user. Defaults to "question".
        context_fields: A list of key names/fields used for the retrieved contexts. Defaults to ["context"].

    Returns:
        Callable[[Log], float]: A function that takes a log as input and returns a score between 0 and 1 indicating
        if the retrieved context is relevant to the query.
    """

    def context_query_relevancy(log: Log) -> float:
        """Quantifies how much the retrieved context relates to the query."""
        question = log.inputs[question_field]
        context = "\n".join(log.inputs[context_field] for context_field in context_fields)

        extracted_sentences = call_openai(
            model="gpt-3.5-turbo-16k",
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
        ).strip()
        if extracted_sentences.lower() == "insufficient information":
            return 0.0
        else:
            n_extracted_sentences = len(sent_tokenize(extracted_sentences))
            n_context_sentences = len(sent_tokenize(context))
            return n_extracted_sentences / n_context_sentences

    return context_query_relevancy
