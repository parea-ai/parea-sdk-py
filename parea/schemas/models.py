from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import json
from enum import Enum

from attrs import define, field, validators

from parea.schemas import EvaluationResult
from parea.schemas.log import EvaluatedLog, LLMInputs


@define
class Completion:
    inference_id: Optional[str] = None
    parent_trace_id: Optional[str] = None
    root_trace_id: Optional[str] = None
    trace_name: Optional[str] = None
    llm_inputs: Optional[Dict[str, Any]] = None
    llm_configuration: LLMInputs = LLMInputs()
    end_user_identifier: Optional[str] = None
    deployment_id: Optional[str] = None
    name: Optional[str] = None
    metadata: Optional[Dict] = None
    tags: Optional[List[str]] = field(factory=list)
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
    llm_inputs: Optional[Dict[str, Any]] = None


@define
class Prompt:
    raw_messages: List[Dict[str, Any]]
    messages: List[Dict[str, Any]]
    inputs: Optional[Dict[str, Any]] = None


@define
class UseDeployedPromptResponse:
    deployment_id: str
    version_number: float
    name: Optional[str] = None
    functions: Optional[List[str]] = None
    function_call: Optional[str] = None
    prompt: Optional[Prompt] = None
    model: Optional[str] = None
    provider: Optional[str] = None
    model_params: Optional[Dict[str, Any]] = None


@define
class FeedbackRequest:
    score: float = field(validator=[validators.ge(0), validators.le(1)])
    trace_id: Optional[str] = None
    name: Optional[str] = None
    target: Optional[str] = None


@define
class TraceLogImage:
    url: str
    caption: Optional[str] = None


@define
class TraceLogCommentSchema:
    comment: str
    user_id: str
    created_at: str


@define
class TraceLogAnnotationSchema:
    created_at: str
    user_id: str
    score: float
    user_email_address: Optional[str] = None
    annotation_name: Optional[str] = None
    value: Optional[str] = None


@define
class TraceLog(EvaluatedLog):
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
    evaluation_metric_names: Optional[List[str]] = field(factory=list)
    apply_eval_frac: float = 1.0
    feedback_score: Optional[float] = None

    # info filled from decorator
    trace_name: Optional[str] = None
    children: List[str] = field(factory=list)

    # metrics filled from either decorator or completion
    end_timestamp: Optional[str] = None
    end_user_identifier: Optional[str] = None
    session_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    tags: Optional[List[str]] = field(factory=list)
    experiment_uuid: Optional[str] = None
    images: Optional[List[TraceLogImage]] = field(factory=list)

    # from UI
    comments: Optional[List[TraceLogCommentSchema]] = None
    annotations: Optional[Dict[int, Dict[str, TraceLogAnnotationSchema]]] = None

    depth: int = 0
    execution_order: int = 0


@define
class TraceLogTree(TraceLog):
    children_logs: Optional[List["TraceLogTree"]] = field(factory=list)


@define
class CacheRequest:
    configuration: LLMInputs = LLMInputs()


@define
class UpdateLog:
    trace_id: str
    field_name_to_value_map: Dict[str, Any]
    root_trace_id: Optional[str] = None


@define
class CreateExperimentRequest:
    name: str
    run_name: str
    metadata: Optional[Dict[str, str]] = None


@define
class ExperimentSchema:
    name: str
    uuid: str
    created_at: str
    metadata: Optional[Dict[str, str]] = None


@define
class EvaluationResultSchema(EvaluationResult):
    id: Optional[int] = None


@define
class TraceStatsSchema:
    trace_id: str
    latency: Optional[float] = 0.0
    input_tokens: Optional[int] = 0
    output_tokens: Optional[int] = 0
    total_tokens: Optional[int] = 0
    cost: Optional[float] = None
    scores: Optional[List[EvaluationResultSchema]] = field(factory=list)


@define
class ExperimentStatsSchema:
    parent_trace_stats: List[TraceStatsSchema] = field(factory=list)

    @property
    def avg_scores(self) -> Dict[str, float]:
        accumulators = {}
        counts = {}
        for trace_stat in self.parent_trace_stats:
            for score in trace_stat.scores:
                accumulators[score.name] = accumulators.get(score.name, 0.0) + score.score
                counts[score.name] = counts.get(score.name, 0) + 1
        return {name: accumulators[name] / counts[name] for name in accumulators}

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
    USAGE: str = "usage"
    OPENAICONFIG: str = "openaiconfig"
    LANGCHAIN_CHILD: str = "langchain_child"


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
    inputs: Dict[str, str] = field(factory=dict)
    target: Optional[str] = None
    tags: List[str] = field(factory=list)


@define
class TestCaseCollection:
    id: int
    name: str
    created_at: str
    last_updated_at: str
    column_names: List[str] = field(factory=list)
    test_cases: Dict[int, TestCase] = field(factory=dict)

    def get_all_test_case_inputs(self) -> Iterable[Dict[str, str]]:
        return (test_case.inputs for test_case in self.test_cases.values())

    def num_test_cases(self) -> int:
        return len(self.test_cases)

    def get_all_test_case_targets(self) -> Iterable[str]:
        return (test_case.target for test_case in self.test_cases.values())

    def get_all_test_inputs_and_targets_tuple(self) -> Iterable[Tuple[Dict[str, str], Optional[str]]]:
        return ((test_case.inputs, test_case.target) for test_case in self.test_cases.values())

    def get_all_test_inputs_and_targets_dict(self) -> Iterable[Dict[str, str]]:
        return ({**test_case.inputs, "target": test_case.target} for test_case in self.test_cases.values())

    def write_to_finetune_jsonl(self, file_path: str):
        """Converts dataset to finetune jsonl format and writes to file_path."""
        jsonl_rows = []
        for inputs, target in self.get_all_test_inputs_and_targets_tuple():
            messages = json.loads(inputs["messages"])
            try:
                function_call = json.loads(target)
                if isinstance(function_call, List):
                    function_call = function_call[0]
                if not "arguments" in function_call:
                    # tool use format, need to convert
                    function_call = function_call["function"]
                function_call["arguments"] = json.dumps(function_call["arguments"])
                assistant_response = {"role": "assistant", "function_call": function_call}
            except json.JSONDecodeError:
                assistant_response = {"role": "assistant", "content": target}
            messages.append(assistant_response)
            converted_row = {"messages": messages}
            if functions := inputs.get("functions", None):
                if loaded_functions := json.loads(functions):
                    converted_row["functions"] = loaded_functions
            jsonl_rows.append(converted_row)

        data = [json.dumps(line) for line in jsonl_rows]
        with open(file_path, "w", encoding="utf-8") as f:
            f.write("\n".join(data))


@define
class CreateTestCase:
    inputs: Dict[str, str]
    target: Optional[str] = None
    tags: List[str] = field(factory=list)


@define
class CreateTestCases:
    id: Optional[int] = None
    name: Optional[str] = None
    test_cases: List[CreateTestCase] = field(factory=list)

    @validators.optional
    def id_or_name_is_set(self, attribute, value):
        if not (self.id or self.name):
            raise ValueError("One of id or name must be set.")


@define
class CreateTestCaseCollection(CreateTestCases):
    # column names excluding reserved names, target and tags
    column_names: List[str] = field(factory=list)


@define
class UpdateTestCase:
    inputs: Optional[Dict[str, Any]] = None
    target: Optional[Union[int, float, str, bool]] = None
    tags: Optional[List[str]] = None


class ExperimentStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@define
class FinishExperimentRequestSchema:
    status: ExperimentStatus
    dataset_level_stats: Optional[List[EvaluationResult]] = field(factory=list)


@define
class ListExperimentUUIDsFilters:
    project_name: Optional[str] = None
    metadata_filter: Optional[Dict[str, Any]] = None
    experiment_name_filter: Optional[str] = None
    run_name_filter: Optional[str] = None
    experiment_uuids: Optional[List[str]] = None


class StatisticOperation(str, Enum):
    MEAN = "mean"
    MEDIAN = "median"
    VARIANCE = "variance"
    STANDARD_DEVIATION = "standard_deviation"
    MIN = "min"
    MAX = "max"
    MSE = "mse"
    MAE = "mae"
    CORRELATION = "correlation"
    SPEARMAN_CORRELATION = "spearman_correlation"
    ACCURACY = "accuracy"
    CUSTOM = "custom"


@define
class ExperimentPinnedStatistic:
    var1: str
    operation: StatisticOperation
    value: float
    var2: Optional[str] = None


@define
class ExperimentWithPinnedStatsSchema:
    name: str
    uuid: str
    created_at: str
    run_name: str
    project_uuid: str
    status: ExperimentStatus
    is_public: bool = False
    metadata: Optional[Dict[str, str]] = None
    pinned_stats: List[ExperimentPinnedStatistic] = []
    num_samples: Optional[int] = None


class FilterOperator(str, Enum):
    EQUALS = "equals"
    NOT_EQUALS = "not_equals"
    LIKE = "like"
    GREATER_THAN_OR_EQUAL = "greater_than_or_equal"
    LESS_THAN_OR_EQUAL = "less_than_or_equal"
    GRATER_THAN = "greater_than"
    LESS_THAN = "less_than"
    IS_NULL = "is_null"
    EXISTS = "exists"
    IN = "in"


@define
class TraceLogFilters:
    filter_field: Optional[str] = None
    filter_operator: Optional[FilterOperator] = None
    filter_value: Optional[str] = None
