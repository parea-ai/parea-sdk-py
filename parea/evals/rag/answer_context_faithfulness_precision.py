from typing import Callable, Optional

from collections import Counter

from parea.schemas.log import Log


def answer_context_faithfulness_precision_factory(context_field: Optional[str] = "context") -> Callable[[Log], float]:
    """Prop. of tokens in model generation which are also present in the retrieved context."""

    def answer_context_faithfulness_precision(log: Log) -> float:
        """Prop. of tokens in model generation which are also present in the retrieved context."""
        context = log.inputs[context_field]

        provider = log.configuration.provider
        model = log.configuration.model

        if provider == "openai":
            import tiktoken

            encoding = tiktoken.encoding_for_model(model)
            context_tokens = encoding.encode(context)
            output_tokens = encoding.encode(log.output)
        else:
            raise NotImplementedError

        if len(context_tokens) == 0:
            return 1.0
        elif len(output_tokens) == 0:
            return 0.0

        common_tokens = Counter(context_tokens) & Counter(output_tokens)
        num_common = sum(common_tokens.values())
        return num_common / len(output_tokens)

    return answer_context_faithfulness_precision
