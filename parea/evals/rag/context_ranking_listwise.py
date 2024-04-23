from typing import Callable, List, Optional

from parea.evals.utils import call_openai, get_context, ndcg
from parea.schemas.log import Log


def context_ranking_listwise_factory(
    question_field: str = "question",
    context_fields: Optional[List[str]] = None,
    ranking_measurement="ndcg",
    n_contexts_to_rank=10,
    model: Optional[str] = "gpt-3.5-turbo-16k",
    is_azure: Optional[bool] = False,
) -> Callable[[Log], float]:
    """
    This factory creates an evaluation function that measures how well the retrieved contexts are ranked by relevancy to the given query
    by listwise estimation of the relevancy of every context to the query. It is based on the paper
    [Zero-Shot Listwise Document Reranking with a Large Language Model](https://arxiv.org/abs/2305.02156) which suggests using an LLM
    to rerank a list of contexts and use that to evaluate how well the contexts are ranked by relevancy to the given query.
    The authors used a progressive listwise reordering if the retrieved contexts don't fit into the context window of the LLM.

    Args:
        is_azure: Whether to use the Azure API. Defaults to False.
        model: The model which should be used for grading. Defaults to "gpt-3.5-turbo-16k".
        question_field (str): The name of the field in the log that contains the question. Defaults to "question".
        context_fields: An optional list of key names/fields used for the retrieved contexts in the input to function. If empty list or None, it will use the output field of the log as context. Defaults to None.
        ranking_measurement (str): The measurement to use for ranking. Defaults to "ndcg".
        n_contexts_to_rank (int): The number of contexts to rank listwise. Defaults to 10.

    Returns:
        Callable[[Log], float]: A function that takes a log as input and returns a score between 0 and 1 indicating
        how well the retrieved context is ranked by their relevancy.

    Raises:
        ValueError: If n_contexts_to_rank is less than 1.
    """
    if n_contexts_to_rank < 1:
        raise ValueError("n_contexts_to_rank must be at least 1.")

    def listwise_reranking(query: str, contexts: List[str]) -> List[int]:
        """Uses a LLM to listwise rerank the contexts. Returns the indices of the contexts in the order of their
        relevance (most relevant to least relevant)."""
        if len(contexts) == 0 or len(contexts) == 1:
            return list(range(len(contexts)))

        prompt = ""
        for i in range(len(contexts)):
            prompt += f"Passage{i + 1} = {contexts[i]}\n"

        prompt += f"""Query = {query}
        Passages = [Passage1, ..., Passage{len(contexts)}]
        Sort the Passages by their relevance to the Query.
        Sorted Passages = ["""

        sorted_list = call_openai(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model=model,
            temperature=0.0,
            is_azure=is_azure,
        )

        s = sorted_list.strip("[] ").replace(" ", "")
        number_strings = s.split(",")
        return [int(num) for num in number_strings if num.isdigit()]

    def progressive_reranking(query: str, contexts: List[str]) -> List[int]:
        """Returns the indices of the contexts in the order of their relevance (most relevant to least relevant)."""
        if len(contexts) <= n_contexts_to_rank:
            return listwise_reranking(query, contexts)

        window_size = n_contexts_to_rank
        window_step = n_contexts_to_rank // 2
        offset = len(contexts) - window_size

        indices = list(range(len(contexts)))

        while offset > 0:
            window_contexts = contexts[offset : offset + window_size]
            window_indices = indices[offset : offset + window_size]
            reranked_indices = listwise_reranking(query, window_contexts)
            contexts[offset : offset + window_size] = [window_contexts[i] for i in reranked_indices]
            indices[offset : offset + window_size] = [window_indices[i] for i in reranked_indices]

            offset -= window_step

        window_contexts = contexts[:window_size]
        window_indices = indices[:window_size]
        reranked_indices = listwise_reranking(query, window_contexts)
        contexts[:window_size] = [window_contexts[i] for i in reranked_indices]
        indices[:window_size] = [window_indices[i] for i in reranked_indices]

        return indices

    def context_ranking(log: Log) -> float:
        """Quantifies if the retrieved context is ranked by their relevancy by re-ranking the contexts."""
        question = log.inputs[question_field]
        contexts = get_context(log, context_fields, True)

        reranked_indices = progressive_reranking(question, contexts)

        if ranking_measurement == "ndcg":
            return ndcg(reranked_indices, list(range(len(contexts))))
        else:
            raise NotImplementedError

    return context_ranking
