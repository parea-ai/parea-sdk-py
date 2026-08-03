import itertools
import json
import math

import pytest

import parea.evals.rag.answer_context_faithfulness_statement_level as statement_level_module
import parea.evals.rag.context_ranking_listwise as listwise_module
import parea.evals.rag.context_ranking_pointwise as pointwise_module
import parea.evals.rag.percent_target_supported_by_context as percent_supported_module
from parea.evals.dataset_level.balanced_acc import balanced_acc_factory
from parea.evals.utils import dcg
from parea.schemas import EvaluatedLog, EvaluationResult
from parea.schemas.log import Log


def patch_openai(monkeypatch, module, responses):
    """Make the module's call_openai return `responses`, one per call, in order."""
    replies = iter(responses)
    monkeypatch.setattr(module, "call_openai", lambda *args, **kwargs: next(replies))


class TestAnswerContextFaithfulnessStatementLevel:
    STATEMENTS = "Paris is the capital of France.\nFrance is in Europe.\nThe Eiffel Tower is in Paris."

    LOG = Log(
        inputs={"question": "Tell me about France.", "context": "Paris is the capital of France. France is in Europe. The Eiffel Tower is in Paris."},
        output="Paris is the capital of France. France is in Europe. The Eiffel Tower is in Paris.",
    )

    @staticmethod
    def nli_response(final_verdicts: str) -> str:
        return (
            "1. Paris is the capital of France.\nExplanation: stated in the context. Verdict: Yes.\n"
            "2. France is in Europe.\nExplanation: stated in the context. Verdict: Yes.\n"
            "3. The Eiffel Tower is in Paris.\nExplanation: stated in the context. Verdict: Yes.\n"
            f"Final verdict for each statement in order: {final_verdicts}"
        )

    @pytest.mark.parametrize(
        "final_verdicts, expected",
        [
            ("Yes. Yes. Yes.", 1.0),
            ("No. No. No.", 0.0),
            ("Yes. No. Yes.", 2 / 3),
            ("No. Yes. No.", 1 / 3),
        ],
    )
    def test_score_is_the_share_of_supported_statements(self, monkeypatch, final_verdicts, expected):
        """A fully grounded answer must score 1.0 and a fully hallucinated one 0.0, not the other way round."""
        patch_openai(monkeypatch, statement_level_module, [self.STATEMENTS, self.nli_response(final_verdicts)])

        score = statement_level_module.answer_context_faithfulness_statement_level_factory()(self.LOG)

        assert score == pytest.approx(expected)

    def test_falls_back_to_the_per_statement_verdicts(self, monkeypatch):
        """Without the summary line the score must come from the grader's verdicts, not the graded answer."""
        verdicts_without_summary = "1. Paris is the capital of France. Verdict: Yes.\n2. France is in Europe. Verdict: Yes.\n3. The Eiffel Tower is in Paris. Verdict: No."
        patch_openai(monkeypatch, statement_level_module, [self.STATEMENTS, verdicts_without_summary])

        score = statement_level_module.answer_context_faithfulness_statement_level_factory()(self.LOG)

        assert score == pytest.approx(2 / 3)

    def test_score_stays_within_the_unit_interval(self, monkeypatch):
        """A grader that returns more verdicts than there are statements must not push the score above 1."""
        patch_openai(monkeypatch, statement_level_module, [self.STATEMENTS, self.nli_response("Yes. Yes. Yes. Yes. Yes.")])

        score = statement_level_module.answer_context_faithfulness_statement_level_factory()(self.LOG)

        assert score == pytest.approx(1.0)


class TestContextRankingListwise:
    CONTEXTS = ["ctx a", "ctx b", "ctx c", "ctx d"]
    LOG = Log(inputs={"question": "What is the capital of France?"}, output=json.dumps(CONTEXTS))

    @staticmethod
    def reranked_reply(permutation) -> str:
        """Format a reranking the way the prompt asks for it: 1-based passage names."""
        return ", ".join(f"Passage{index + 1}" for index in permutation) + "]"

    def score_for(self, monkeypatch, permutation) -> float:
        patch_openai(monkeypatch, listwise_module, [self.reranked_reply(permutation)])
        return listwise_module.context_ranking_listwise_factory()(self.LOG)

    def test_agreeing_with_the_retrieved_order_scores_one(self, monkeypatch):
        """If the reranker leaves the retrieved order untouched, the retrieval was perfectly ranked."""
        assert self.score_for(monkeypatch, range(len(self.CONTEXTS))) == pytest.approx(1.0)

    def test_reversing_the_retrieved_order_scores_lowest(self, monkeypatch):
        """The reversed order is the worst possible ranking and must not score better than the perfect one."""
        perfect = self.score_for(monkeypatch, range(len(self.CONTEXTS)))
        reversed_order = self.score_for(monkeypatch, reversed(range(len(self.CONTEXTS))))

        assert reversed_order < perfect

    def test_score_decreases_as_the_ranking_gets_worse(self, monkeypatch):
        """Guards against the metric being inverted: the identity ranking must be the unique maximum."""
        scores = {permutation: self.score_for(monkeypatch, permutation) for permutation in itertools.permutations(range(len(self.CONTEXTS)))}

        identity = tuple(range(len(self.CONTEXTS)))
        assert max(scores, key=scores.get) == identity
        assert min(scores, key=scores.get) == identity[::-1]
        assert all(0.0 <= score <= 1.0 for score in scores.values())

    @pytest.mark.parametrize(
        "reply",
        [
            "Passage2, Passage1, Passage3, Passage4]",
            "2, 1, 3, 4]",
            "[Passage2, Passage1, Passage3, Passage4]",
            "Sorted Passages = [Passage2, Passage1, Passage3, Passage4]",
        ],
    )
    def test_reply_formats_are_parsed_equivalently(self, monkeypatch, reply):
        """All of these encode the same 1-based ranking, so they must produce the same score."""
        patch_openai(monkeypatch, listwise_module, [reply])
        score = listwise_module.context_ranking_listwise_factory()(self.LOG)

        assert score == pytest.approx(self.score_for(monkeypatch, [1, 0, 2, 3]))

    def test_unusable_reply_still_yields_a_valid_score(self, monkeypatch):
        """A reply with no ranking in it must not drop contexts or raise; it just leaves the order as retrieved."""
        patch_openai(monkeypatch, listwise_module, ["I'm sorry, I cannot help with that."])

        score = listwise_module.context_ranking_listwise_factory()(self.LOG)

        assert score == pytest.approx(1.0)

    def test_empty_context_list_scores_zero(self, monkeypatch):
        patch_openai(monkeypatch, listwise_module, [])

        score = listwise_module.context_ranking_listwise_factory()(Log(inputs={"question": "q"}, output=json.dumps([])))

        assert score == 0.0

    def test_ranking_a_single_context_at_a_time_terminates(self, monkeypatch):
        """n_contexts_to_rank=1 used to make the sliding window step by 0 and loop forever."""
        patch_openai(monkeypatch, listwise_module, ["1]"] * 10)

        score = listwise_module.context_ranking_listwise_factory(n_contexts_to_rank=1)(self.LOG)

        assert 0.0 <= score <= 1.0


class TestContextRankingPointwise:
    LOG = Log(inputs={"question": "What is the capital of France?"}, output=json.dumps(["ctx a", "ctx b", "ctx c"]))

    @pytest.mark.parametrize(
        "verdicts, expected",
        [
            (["Yes", "Yes", "Yes"], 1.0),
            (["No", "No", "No"], 0.0),
            (["Yes", "No", "Yes"], (1 / 1 + 2 / 3) / 2),
        ],
    )
    def test_average_precision_matches_the_manual_calculation(self, monkeypatch, verdicts, expected):
        patch_openai(monkeypatch, pointwise_module, [json.dumps({"reason": "...", "verdict": verdict}) for verdict in verdicts])

        score = pointwise_module.context_ranking_pointwise_factory()(self.LOG)

        assert score == pytest.approx(expected)

    @pytest.mark.parametrize(
        "malformed",
        [
            "I cannot answer that.",
            json.dumps({"reason": "unsure"}),
            "",
        ],
    )
    def test_a_malformed_verdict_does_not_produce_nan(self, monkeypatch, malformed):
        """One unparseable response used to turn the entire score into NaN."""
        patch_openai(monkeypatch, pointwise_module, [json.dumps({"verdict": "Yes"}), malformed, json.dumps({"verdict": "Yes"})])

        score = pointwise_module.context_ranking_pointwise_factory()(self.LOG)

        assert not math.isnan(score)
        assert 0.0 <= score <= 1.0


class TestPercentTargetSupportedByContext:
    LOG = Log(inputs={"question": "Tell me about Einstein."}, output="Einstein was a physicist.", target="Einstein was a physicist. He won a Nobel Prize.")

    @staticmethod
    def classification(*attributions) -> str:
        items = [{f"statement_{i + 1}": "...", "reason": "...", "Attributed": attributed} for i, attributed in enumerate(attributions)]
        return json.dumps(items, indent=4)

    @pytest.mark.parametrize(
        "attributions, expected",
        [
            (("Yes", "Yes"), 1.0),
            (("No", "No"), 0.0),
            (("Yes", "No", "Yes"), 2 / 3),
        ],
    )
    def test_score_is_the_share_of_attributed_statements(self, monkeypatch, attributions, expected):
        patch_openai(monkeypatch, percent_supported_module, [self.classification(*attributions)])

        score = percent_supported_module.percent_target_supported_by_context_factory()(self.LOG)

        assert score == pytest.approx(expected)

    def test_json_wrapped_in_prose_is_parsed(self, monkeypatch):
        """Models routinely wrap their answer in a code fence, which used to raise inside eval()."""
        patch_openai(monkeypatch, percent_supported_module, [f"Here is the classification:\n```json\n{self.classification('Yes', 'No')}\n```"])

        score = percent_supported_module.percent_target_supported_by_context_factory()(self.LOG)

        assert score == pytest.approx(0.5)

    def test_model_output_is_never_executed(self, monkeypatch, tmp_path):
        """The classification used to be passed to eval(), so the grader could run arbitrary code."""
        canary = tmp_path / "canary.txt"
        payload = f'[{{"statement_1": "...", "Attributed": "Yes"}}] and __import__("pathlib").Path({str(canary)!r}).write_text("executed")'
        patch_openai(monkeypatch, percent_supported_module, [payload])

        percent_supported_module.percent_target_supported_by_context_factory()(self.LOG)

        assert not canary.exists()

    def test_unparseable_classification_returns_no_score(self, monkeypatch):
        """Reporting 0.0 here is indistinguishable from 'nothing was supported', so return no score at all."""
        patch_openai(monkeypatch, percent_supported_module, ["I'm sorry, I cannot help with that."])

        with pytest.warns(UserWarning):
            score = percent_supported_module.percent_target_supported_by_context_factory()(self.LOG)

        assert score is None

    def test_missing_target_returns_no_score(self, monkeypatch):
        patch_openai(monkeypatch, percent_supported_module, [])

        score = percent_supported_module.percent_target_supported_by_context_factory()(Log(inputs={"question": "q"}, output="out"))

        assert score is None


class TestBalancedAccuracy:
    def test_fractional_scores_are_thresholded_not_truncated(self):
        """int(0.9) is 0, which used to report a class with near-perfect scores as entirely wrong."""
        logs = [
            EvaluatedLog(target="a", scores=[EvaluationResult(name="correctness", score=0.9)]),
            EvaluatedLog(target="a", scores=[EvaluationResult(name="correctness", score=0.8)]),
            EvaluatedLog(target="b", scores=[EvaluationResult(name="correctness", score=1.0)]),
        ]

        result = balanced_acc_factory("correctness")(logs)

        assert result.score == pytest.approx(1.0)

    def test_recall_is_averaged_over_classes(self):
        """Two of three 'a' logs are correct and the single 'b' log is wrong: (2/3 + 0) / 2."""
        logs = [
            EvaluatedLog(target="a", scores=[EvaluationResult(name="correctness", score=1.0)]),
            EvaluatedLog(target="a", scores=[EvaluationResult(name="correctness", score=1.0)]),
            EvaluatedLog(target="a", scores=[EvaluationResult(name="correctness", score=0.0)]),
            EvaluatedLog(target="b", scores=[EvaluationResult(name="correctness", score=0.0)]),
        ]

        result = balanced_acc_factory("correctness")(logs)

        assert result.score == pytest.approx((2 / 3) / 2)

    def test_no_matching_scores_returns_none(self):
        logs = [EvaluatedLog(target="a", scores=[EvaluationResult(name="other", score=1.0)])]

        assert balanced_acc_factory("correctness")(logs) is None


class TestDCG:
    @pytest.mark.parametrize("relevance", [1, 32, 63, 64, 70, 200])
    def test_gain_grows_with_relevance(self, relevance):
        """2**rel was computed on an int64 array, so a relevance grade above 62 silently wrapped to 0."""
        gain = dcg([relevance], [0])

        assert math.isfinite(gain)
        assert gain > dcg([relevance - 1], [0])

    def test_a_well_ranked_list_beats_a_badly_ranked_one(self):
        n_contexts = 70
        ranking = list(range(n_contexts))

        best_first = dcg(list(range(n_contexts, 0, -1)), ranking)
        best_last = dcg(list(range(1, n_contexts + 1)), ranking)

        assert math.isfinite(best_first)
        assert best_first > best_last
