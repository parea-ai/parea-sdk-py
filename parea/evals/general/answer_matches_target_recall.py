from collections import Counter

from parea.schemas.log import Log


def answer_matches_target_recall(log: Log) -> float:
    """Prop. of tokens in target/reference answer which are also in model generation."""
    target = log.target
    output = log.output

    provider = log.configuration.provider
    model = log.configuration.model

    if provider == "openai":
        import tiktoken

        encoding = tiktoken.encoding_for_model(model)
        target_tokens = encoding.encode(target)
        output_tokens = encoding.encode(output)
    else:
        raise NotImplementedError

    if len(target_tokens) == 0:
        return 1.0
    common_tokens = Counter(target_tokens) & Counter(output_tokens)
    num_common = sum(common_tokens.values())
    return num_common / len(target_tokens)
