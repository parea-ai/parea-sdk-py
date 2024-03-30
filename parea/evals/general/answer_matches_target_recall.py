from typing import Union

from collections import Counter

from parea.evals.utils import get_tokens
from parea.schemas.log import Log


def answer_matches_target_recall(log: Log) -> Union[float, None]:
    """Prop. of tokens in target/reference answer which are also in model generation."""
    if (target := log.target) is None:
        return None
    output = log.output
    model = log.configuration.model

    target_tokens = get_tokens(model, target)
    output_tokens = get_tokens(model, output)

    if len(target_tokens) == 0:
        return 1.0
    common_tokens = Counter(target_tokens) & Counter(output_tokens)
    num_common = sum(common_tokens.values())
    return num_common / len(target_tokens)
