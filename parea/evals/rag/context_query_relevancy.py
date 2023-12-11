from typing import Callable, List

from parea.evals.utils import call_openai, sent_tokenize
from parea.schemas.log import Log


def context_query_relevancy_factory(question_field: str = "question", context_fields: List[str] = ["context"]) -> Callable[[Log], float]:
    """Quantifies how much the retrieved context relates to the query."""

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
