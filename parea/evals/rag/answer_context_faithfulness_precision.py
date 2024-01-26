from typing import Callable, Optional

from collections import Counter

from parea.evals.utils import get_tokens
from parea.schemas.log import Log


def answer_context_faithfulness_precision_factory(context_field: Optional[str] = "context") -> Callable[[Log], float]:
    """
        This factory creates an evaluation function that calculates the how many tokens in the generated answer are also present in the retrieved context.
    It is based on the paper [Evaluating Correctness and Faithfulness of Instruction-Following Models for Question Answering](https://arxiv.org/abs/2307.16877)
    which finds that this method only slightly lags behind GPT-4 and outperforms GPT-3.5-turbo (see Table 4 from the above paper).

        Args:
            context_field: The key name/field used for the retrieved context. Defaults to "context".

        Returns:
            Callable[[Log], float]: A function that takes a log as input and returns a score between 0 and 1 indicating
            how many tokens in the generated answer are also present in the retrieved context.
    """

    def answer_context_faithfulness_precision(log: Log) -> float:
        """Prop. of tokens in model generation which are also present in the retrieved context."""
        context = log.inputs[context_field]
        model = log.configuration.model

        context_tokens = get_tokens(model, context)
        output_tokens = get_tokens(model, log.output)

        if len(context_tokens) == 0:
            return 1.0
        elif len(output_tokens) == 0:
            return 0.0

        common_tokens = Counter(context_tokens) & Counter(output_tokens)
        num_common = sum(common_tokens.values())
        return num_common / len(output_tokens)

    return answer_context_faithfulness_precision
