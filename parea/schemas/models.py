from typing import Any, Iterable, Optional

from enum import Enum

from attrs import define, field, validators

from parea.schemas.log import LLMInputs, Log


@define
class Completion:
    inference_id: Optional[str] = None
    parent_trace_id: Optional[str] = None
    root_trace_id: Optional[str] = None
    trace_name: Optional[str] = None
    llm_inputs: Optional[dict[str, Any]] = None
    llm_configuration: LLMInputs = LLMInputs()
    end_user_identifier: Optional[str] = None
    deployment_id: Optional[str] = None
    name: Optional[str] = None
    metadata: Optional[dict] = None
    tags: Optional[list[str]] = field(factory=list)
    target: Optional[str] = None
    cache: bool = True
    log_omit_inputs: bool = False
    log_omit_outputs: bool = False
    log_omit: bool = False
    experiment_uuid: Optional[str] = None
    project_uuid: str = "default"


@define
class CompletionResponse:
    inference_id: str
    content: str
    latency: float
    input_tokens: int
    output_tokens: int
    total_tokens: int
    cost: float
    model: str
    provider: str
    cache_hit: bool
    status: str
    start_timestamp: str
    end_timestamp: str
    error: Optional[str] = None


@define
class UseDeployedPrompt:
    deployment_id: str
    llm_inputs: Optional[dict[str, Any]] = None


@define
class Prompt:
    raw_messages: list[dict[str, Any]]
    messages: list[dict[str, Any]]
    inputs: Optional[dict[str, Any]] = None


@define
class UseDeployedPromptResponse:
    deployment_id: str
    name: Optional[str] = None
    functions: Optional[list[str]] = None
    function_call: Optional[str] = None
    prompt: Optional[Prompt] = None
    model: Optional[str] = None
    provider: Optional[str] = None
    model_params: Optional[dict[str, Any]] = None


@define
class FeedbackRequest:
    score: float = field(validator=[validators.ge(0), validators.le(1)])
    trace_id: Optional[str] = None
    inference_id: Optional[str] = None
    name: Optional[str] = None
    target: Optional[str] = None


@define
class NamedEvaluationScore:
    name: str
    score: float


@define
class TraceLog(Log):
    trace_id: Optional[str] = field(default=None, validator=validators.instance_of(str))
    parent_trace_id: Optional[str] = field(default=None, validator=validators.instance_of(str))
    root_trace_id: Optional[str] = field(default=None, validator=validators.instance_of(str))
    start_timestamp: Optional[str] = field(default=None, validator=validators.instance_of(str))
    organization_id: Optional[str] = None
    project_uuid: Optional[str] = None

    # metrics filled from completion
    error: Optional[str] = None
    status: Optional[str] = None
    deployment_id: Optional[str] = None
    cache_hit: bool = False
    output_for_eval_metrics: Optional[str] = None
    evaluation_metric_names: Optional[list[str]] = field(factory=list)
    scores: Optional[list[NamedEvaluationScore]] = field(factory=list)
    apply_eval_frac: Optional[float] = 1.0
    feedback_score: Optional[float] = None

    # info filled from decorator
    trace_name: Optional[str] = None
    children: list[str] = field(factory=list)

    # metrics filled from either decorator or completion
    end_timestamp: Optional[str] = None
    end_user_identifier: Optional[str] = None
    metadata: Optional[dict[str, Any]] = None
    tags: Optional[list[str]] = field(factory=list)
    experiment_uuid: Optional[str] = None


@define
class TraceLogTree(TraceLog):
    children: Optional[list[TraceLog]] = field(factory=list)


@define
class CacheRequest:
    configuration: LLMInputs = LLMInputs()


@define
class UpdateLog:
    trace_id: str
    field_name_to_value_map: dict[str, Any]


@define
class CreateExperimentRequest:
    name: str
    metadata: Optional[dict[str, str]] = None


@define
class ExperimentSchema:
    name: str
    uuid: str
    created_at: str
    metadata: Optional[dict[str, str]] = None


@define
class EvaluationScoreSchema(NamedEvaluationScore):
    id: Optional[int] = None


@define
class TraceStatsSchema:
    trace_id: str
    latency: Optional[float] = 0.0
    input_tokens: Optional[int] = 0
    output_tokens: Optional[int] = 0
    total_tokens: Optional[int] = 0
    cost: Optional[float] = None
    scores: Optional[list[EvaluationScoreSchema]] = field(factory=list)


@define
class ExperimentStatsSchema:
    parent_trace_stats: list[TraceStatsSchema] = field(factory=list)

    def cumulative_avg_score(self) -> float:
        """Returns the average score across all evals."""
        scores = [score.score for trace_stat in self.parent_trace_stats for score in trace_stat.scores]
        return sum(scores) / len(scores) if scores else 0.0

    def avg_score(self, score_name: str) -> float:
        """Returns the average score for a given eval."""
        scores = [score.score for trace_stat in self.parent_trace_stats for score in trace_stat.scores if score.name == score_name]
        return sum(scores) / len(scores) if scores else 0.0


class UpdateTraceScenario(str, Enum):
    RESULT: str = "result"
    ERROR: str = "error"
    CHAIN: str = "chain"


@define
class CreateGetProjectSchema:
    name: str


@define
class ProjectSchema(CreateGetProjectSchema):
    uuid: str
    created_at: str


@define
class CreateGetProjectResponseSchema(ProjectSchema):
    was_created: bool


@define
class TestCase:
    id: int
    test_case_collection_id: int
    inputs: dict[str, str] = field(factory=dict)
    target: Optional[str] = None
    tags: list[str] = field(factory=list)


@define
class TestCaseCollection:
    id: int
    name: str
    created_at: str
    last_updated_at: str
    column_names: list[str] = field(factory=list)
    test_cases: dict[int, TestCase] = field(factory=dict)

    def get_all_test_case_inputs(self) -> Iterable[dict[str, str]]:
        return (test_case.inputs for test_case in self.test_cases.values())

    def num_test_cases(self) -> int:
        return len(self.test_cases)

    def get_all_test_case_targets(self) -> Iterable[str]:
        return (test_case.target for test_case in self.test_cases.values())

    def get_all_test_inputs_and_targets_tuple(self) -> Iterable[tuple[dict[str, str], Optional[str]]]:
        return ((test_case.inputs, test_case.target) for test_case in self.test_cases.values())

    def get_all_test_inputs_and_targets_dict(self) -> Iterable[dict[str, str]]:
        return ({**test_case.inputs, "target": test_case.target} for test_case in self.test_cases.values())


@define
class CreateTestCase:
    inputs: dict[str, str]
    target: Optional[str] = None
    tags: list[str] = field(factory=list)


@define
class CreateTestCases:
    name: str
    test_cases: list[CreateTestCase] = field(factory=list)


@define
class CreateTestCaseCollection(CreateTestCases):
    # column names excluding reserved names, target and tags
    column_names: list[str] = field(factory=list)
