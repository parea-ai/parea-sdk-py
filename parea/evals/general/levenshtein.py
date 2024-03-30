from typing import Union

from Levenshtein import distance

from parea.schemas import Log


def levenshtein(log: Log) -> Union[float, None]:
    output = log.output
    if (target := log.target) is None:
        return None

    output, target = str(output), str(target)
    max_len = max(len(x) for x in [output, target])

    score = 1
    if max_len > 0:
        score = 1 - (distance(output, target) / max_len)

    return score
