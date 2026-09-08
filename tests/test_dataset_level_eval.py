import pytest

from parea.evals.dataset_level.balanced_acc import balanced_acc_factory
from parea.experiment.experiment import apply_dataset_eval
from parea.schemas import EvaluationResult
from parea.schemas.models import TraceLog
from parea.utils.trace_utils import trace_data


def _trace(trace_id, experiment_uuid, *, root_trace_id=None, target="yes", score=1.0, score_name="is_correct"):
    root = root_trace_id or trace_id
    return TraceLog(
        trace_id=trace_id,
        parent_trace_id=root,
        root_trace_id=root,
        start_timestamp="2024-01-01T00:00:00",
        experiment_uuid=experiment_uuid,
        target=target,
        scores=[EvaluationResult(name=score_name, score=score)] if score is not None else [],
    )


@pytest.fixture
def isolated_trace_store():
    token = trace_data.set({})
    try:
        yield trace_data.get()
    finally:
        trace_data.reset(token)


def test_dataset_eval_only_receives_root_traces_from_the_current_experiment(isolated_trace_store):
    isolated_trace_store["prev"] = _trace("prev", "exp-1")
    isolated_trace_store["curr"] = _trace("curr", "exp-2")
    isolated_trace_store["child"] = _trace("child", "exp-2", root_trace_id="curr")
    isolated_trace_store["unrelated"] = _trace("unrelated", None)

    seen = []

    def capture(logs):
        seen.append([log.trace_id for log in logs])
        return 1.0

    apply_dataset_eval([capture], "exp-2")

    assert seen == [["curr"]]


def test_dataset_eval_score_is_not_contaminated_by_prior_experiment(isolated_trace_store):
    # Prior experiment was perfect. The current experiment failed every class.
    # Mixing the two would yield balanced accuracy 0.5 instead of 0.0.
    isolated_trace_store["old_yes"] = _trace("old_yes", "exp-1", target="yes", score=1.0)
    isolated_trace_store["old_no"] = _trace("old_no", "exp-1", target="no", score=1.0)
    isolated_trace_store["new_yes"] = _trace("new_yes", "exp-2", target="yes", score=0.0)
    isolated_trace_store["new_no"] = _trace("new_no", "exp-2", target="no", score=0.0)

    results = apply_dataset_eval([balanced_acc_factory("is_correct")], "exp-2")

    assert len(results) == 1
    assert results[0].name == "balanced_acc_is_correct"
    assert results[0].score == 0.0


def test_dataset_eval_with_no_matching_traces_does_not_use_other_experiments(isolated_trace_store):
    isolated_trace_store["prev"] = _trace("prev", "exp-1")

    seen = []

    def capture(logs):
        seen.append(len(logs))
        return None

    results = apply_dataset_eval([capture], "exp-2")

    assert seen == [0]
    assert results == []
