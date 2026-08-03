from typing import List, Union

from collections import defaultdict

from parea.schemas import EvaluatedLog, EvaluationResult


def balanced_acc_factory(score_name: str):
    def balanced_acc(logs: List[EvaluatedLog]) -> Union[EvaluationResult, None]:
        correct = defaultdict(int)
        total = defaultdict(int)
        for log in logs:
            if (eval_result := log.get_score(score_name)) is not None:
                # Threshold instead of truncating: int(0.9) is 0, which would report a class as
                # entirely wrong even though every one of its scores was almost perfect.
                correct[log.target] += int(eval_result.score >= 0.5)
                total[log.target] += 1
        recalls = [correct[key] / total[key] for key in correct]

        if len(recalls) == 0:
            return None

        return EvaluationResult(name=f"balanced_acc_{score_name}", score=sum(recalls) / len(recalls))

    return balanced_acc
