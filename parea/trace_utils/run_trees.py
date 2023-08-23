from typing import Any, Dict, List, Optional, cast

import logging
import os
from concurrent.futures import Future, ThreadPoolExecutor, wait
from datetime import datetime
from enum import Enum
from uuid import UUID, uuid4

from parea.client_two import ID_TYPE, Client, get_runtime_environment

try:
    from pydantic.v1 import BaseModel, Field, PrivateAttr, root_validator, validator  # type: ignore[import]
except ImportError:
    from pydantic import BaseModel, Field, PrivateAttr, StrictBool, StrictFloat, StrictInt, root_validator, validator

logger = logging.getLogger(__name__)


class RunTypeEnum(str, Enum):
    """(Deprecated) Enum for run types. Use string directly."""

    tool = "tool"
    chain = "chain"
    llm = "llm"
    retriever = "retriever"
    embedding = "embedding"
    prompt = "prompt"
    parser = "parser"


class RunBase(BaseModel):
    """Base Run schema."""

    id: UUID
    name: str
    start_time: datetime
    run_type: str
    """The type of run, such as tool, chain, llm, retriever,
    embedding, prompt, parser."""
    end_time: Optional[datetime] = None
    extra: Optional[dict] = None
    error: Optional[str] = None
    serialized: Optional[dict]
    events: Optional[list[dict]] = None
    inputs: dict
    outputs: Optional[dict] = None
    reference_example_id: Optional[UUID] = None
    parent_run_id: Optional[UUID] = None
    tags: Optional[list[str]] = None


class Run(RunBase):
    """Run schema when loading from the DB."""

    execution_order: int
    """The execution order of the run within a run trace."""
    session_id: Optional[UUID] = None
    """The project ID this run belongs to."""
    child_run_ids: Optional[list[UUID]] = None
    """The child run IDs of this run."""
    child_runs: Optional[list["Run"]] = None
    """The child runs of this run, if instructed to load using the client
    These are not populated by default, as it is a heavier query to make."""
    feedback_stats: Optional[dict[str, Any]] = None
    """Feedback stats for this run."""
    app_path: Optional[str] = None
    """Relative URL path of this run within the app."""
    _host_url: Optional[str] = PrivateAttr(default=None)

    def __init__(self, _host_url: Optional[str] = None, **kwargs: Any) -> None:
        """Initialize a Run object."""
        super().__init__(**kwargs)
        self._host_url = _host_url

    @property
    def url(self) -> Optional[str]:
        """URL of this run within the app."""
        if self._host_url and self.app_path:
            return f"{self._host_url}{self.app_path}"
        return None


def _make_thread_pool() -> ThreadPoolExecutor:
    """Ensure a thread pool exists in the current context."""
    return ThreadPoolExecutor(max_workers=1)


class RunTree(RunBase):
    """Run Schema with back-references for posting runs."""

    name: str
    id: UUID = Field(default_factory=uuid4)
    start_time: datetime = Field(default_factory=datetime.utcnow)
    parent_run: Optional["RunTree"] = Field(default=None, exclude=True)
    child_runs: list["RunTree"] = Field(
        default_factory=list,
        exclude={"__all__": {"parent_run_id"}},
    )
    session_name: str = Field(
        default_factory=lambda: os.environ.get(
            # TODO: Deprecate PAREA_SESSION
            "PAREA_PROJECT",
            os.environ.get("PAREA_SESSION", "default"),
        ),
        alias="project_name",
    )
    session_id: Optional[UUID] = Field(default=None, alias="project_id")
    execution_order: int = 1
    child_execution_order: int = Field(default=1, exclude=True)
    extra: dict = Field(default_factory=dict)
    client: Client = Field(default_factory=Client, exclude=True)
    executor: ThreadPoolExecutor = Field(default_factory=_make_thread_pool, exclude=True)
    _futures: list[Future] = PrivateAttr(default_factory=list)

    class Config:
        arbitrary_types_allowed = True
        allow_population_by_field_name = True

    @validator("executor", pre=True)
    def validate_executor(cls, v: Optional[ThreadPoolExecutor]) -> ThreadPoolExecutor:
        """Ensure the executor is running."""
        if v is None:
            return _make_thread_pool()
        if v._shutdown:
            raise ValueError("Executor has been shutdown.")
        return v

    @root_validator(pre=True)
    def infer_defaults(cls, values: dict) -> dict:
        """Assign name to the run."""
        if "serialized" not in values:
            values["serialized"] = {"name": values["name"]}
        if "execution_order" not in values:
            values["execution_order"] = 1
        if "child_execution_order" not in values:
            values["child_execution_order"] = values["execution_order"]
        if values.get("parent_run") is not None:
            values["parent_run_id"] = values["parent_run"].id
        extra = cast(dict, values.setdefault("extra", {}))
        runtime = cast(dict, extra.setdefault("runtime", {}))
        runtime.update(get_runtime_environment())
        return values

    def end(
        self,
        *,
        outputs: Optional[dict] = None,
        error: Optional[str] = None,
        end_time: Optional[datetime] = None,
    ) -> None:
        """Set the end time of the run and all child runs."""
        self.end_time = end_time or datetime.utcnow()
        if outputs is not None:
            self.outputs = outputs
        if error is not None:
            self.error = error
        if self.parent_run:
            self.parent_run.child_execution_order = max(
                self.parent_run.child_execution_order,
                self.child_execution_order,
            )

    def create_child(
        self,
        name: str,
        run_type: str,
        *,
        run_id: Optional[ID_TYPE] = None,
        serialized: Optional[dict] = None,
        inputs: Optional[dict] = None,
        outputs: Optional[dict] = None,
        error: Optional[str] = None,
        reference_example_id: Optional[UUID] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        tags: Optional[list[str]] = None,
        extra: Optional[dict] = None,
    ) -> "RunTree":
        """Add a child run to the run tree."""
        execution_order = self.child_execution_order + 1
        serialized_ = serialized or {"name": name}
        run = RunTree(
            name=name,
            id=run_id or uuid4(),
            serialized=serialized_,
            inputs=inputs or {},
            outputs=outputs or {},
            error=error,
            run_type=run_type,
            reference_example_id=reference_example_id,
            start_time=start_time or datetime.utcnow(),
            end_time=end_time,
            execution_order=execution_order,
            child_execution_order=execution_order,
            extra=extra or {},
            parent_run=self,
            session_name=self.session_name,
            client=self.client,
            executor=self.executor,
            tags=tags,
        )
        self.child_runs.append(run)
        return run

    def post(self, exclude_child_runs: bool = True) -> Future:
        """Post the run tree to the API asynchronously."""
        exclude = {"child_runs"} if exclude_child_runs else None
        kwargs = self.dict(exclude=exclude, exclude_none=True)
        self._futures.append(
            self.executor.submit(
                self.client.create_run,
                **kwargs,
            )
        )
        return self._futures[-1]

    def patch(self) -> Future:
        """Patch the run tree to the API in a background thread."""
        self._futures.append(
            self.executor.submit(
                self.client.update_run,
                run_id=self.id,
                outputs=self.outputs.copy() if self.outputs else None,
                error=self.error,
                parent_run_id=self.parent_run_id,
                reference_example_id=self.reference_example_id,
            )
        )
        return self._futures[-1]

    def wait(self) -> None:
        """Wait for all _futures to complete."""
        futures = self._futures
        wait(self._futures)
        for future in futures:
            self._futures.remove(future)


# import logging
# import os
# from concurrent.futures import Future, ThreadPoolExecutor, as_completed
# from dataclasses import dataclass, field
# from datetime import datetime
# from enum import Enum
# from functools import lru_cache
# from typing import Dict, List, Optional, Any
# from uuid import uuid4
#
# from parea.client_two import Client, get_runtime_environment, ID_TYPE
#
# logger = logging.getLogger(__name__)
#
# # Using Enum for run types. It prevents invalid values and improves code readability.
# class RunTypeEnum(str, Enum):
#     tool = "tool"
#     chain = "chain"
#     llm = "llm"
#     retriever = "retriever"
#     embedding = "embedding"
#     prompt = "prompt"
#     parser = "parser"
#
#
# @dataclass
# class RunBase:
#     """Base Run schema."""
#
#     id: str
#     name: str
#     start_time: datetime
#     run_type: RunTypeEnum
#     end_time: Optional[datetime] = None
#     extra: Optional[dict] = None
#     error: Optional[str] = None
#     serialized: Optional[dict] = None
#     events: Optional[List[Dict]] = None
#     inputs: dict = field(default_factory=dict)
#     outputs: Optional[dict] = None
#     reference_example_id: Optional[str] = None
#     parent_run_id: Optional[str] = None
#     tags: Optional[List[str]] = None
#
#
# @dataclass
# class Run(RunBase):
#     """Run schema when loading from the DB."""
#
#     execution_order: Optional[int] = None
#     session_id: Optional[str] = None
#     child_run_ids: Optional[List[str]] = None
#     child_runs: Optional[List["Run"]] = None
#     feedback_stats: Optional[Dict[str, Any]] = None
#     app_path: Optional[str] = None
#     _host_url: Optional[str] = None
#
#     @property
#     def url(self) -> Optional[str]:
#         """URL of this run within the app."""
#         if self._host_url and self.app_path:
#             return f"{self._host_url}{self.app_path}"
#         return None
#
#
# @lru_cache
# def _make_thread_pool() -> ThreadPoolExecutor:
#     """Ensure a thread pool exists in the current context."""
#     return ThreadPoolExecutor(max_workers=1)
#
#
# @dataclass
# class RunTree(RunBase):
#     """Run Schema with back-references for posting runs."""
#
#     parent_run: Optional["RunTree"] = None
#     child_runs: List["RunTree"] = field(default_factory=list)
#     session_name: str = field(default_factory=lambda: os.environ.get("PAREA_PROJECT", os.environ.get("PAREA_SESSION", "default")))
#     session_id: Optional[str] = None
#     execution_order: int = 1
#     child_execution_order: int = 1
#     extra: Dict = field(default_factory=dict)
#     client: Client = field(default_factory=Client)
#     executor: ThreadPoolExecutor = field(default_factory=_make_thread_pool)
#     _futures: List[Future] = field(default_factory=list)
#
#     def __post_init__(self):
#         """Assign name to the run."""
#         if self.serialized is None:
#             self.serialized = {"name": self.name}
#         if self.extra is None:
#             self.extra = {}
#         runtime = self.extra.setdefault("runtime", {})
#         runtime.update(get_runtime_environment())
#
#     def end(
#         self,
#         *,
#         outputs: Optional[Dict] = None,
#         error: Optional[str] = None,
#         end_time: Optional[datetime] = None,
#     ) -> None:
#         """Set the end time of the run and all child runs."""
#         self.end_time = end_time or datetime.utcnow()
#         if outputs is not None:
#             self.outputs = outputs
#         if error is not None:
#             self.error = error
#         if self.parent_run:
#             self.parent_run.child_execution_order = max(
#                 self.parent_run.child_execution_order,
#                 self.child_execution_order,
#             )
#
#     def create_child(
#         self,
#         name: str,
#         run_type: RunTypeEnum,
#         *,
#         run_id: Optional[ID_TYPE] = None,
#         serialized: Optional[Dict] = None,
#         inputs: Optional[Dict] = None,
#         outputs: Optional[Dict] = None,
#         error: Optional[str] = None,
#         reference_example_id: Optional[str] = None,
#         start_time: Optional[datetime] = None,
#         end_time: Optional[datetime] = None,
#         tags: Optional[List[str]] = None,
#         extra: Optional[Dict] = None,
#     ) -> "RunTree":
#         """Add a child run to the run tree."""
#         execution_order = self.child_execution_order + 1
#         serialized_ = serialized or {"name": name}
#         run = RunTree(
#             name=name,
#             id=run_id or str(uuid4()),
#             serialized=serialized_,
#             inputs=inputs or {},
#             outputs=outputs or {},
#             error=error,
#             run_type=run_type,
#             reference_example_id=reference_example_id,
#             start_time=start_time or datetime.utcnow(),
#             end_time=end_time,
#             execution_order=execution_order,
#             child_execution_order=execution_order,
#             extra=extra or {},
#             parent_run=self,
#             session_name=self.session_name,
#             client=self.client,
#             executor=self.executor,
#             tags=tags,
#         )
#         self.child_runs.append(run)
#         return run
#
#     def post(self, exclude_child_runs: bool = True) -> Future:
#         """Post the run tree to the API asynchronously."""
#         self._futures.append(
#             self.executor.submit(
#                 self.client.create_run,
#                 **self.__dict__,
#             )
#         )
#         return self._futures[-1]
#
#     def patch(self) -> Future:
#         """Patch the run tree to the API in a background thread."""
#         self._futures.append(
#             self.executor.submit(
#                 self.client.update_run,
#                 run_id=self.id,
#                 outputs=self.outputs,
#                 error=self.error,
#                 parent_run_id=self.parent_run_id,
#                 reference_example_id=self.reference_example_id,
#             )
#         )
#         return self._futures[-1]
#
#     def wait(self) -> None:
#         """Wait for all _futures to complete."""
#         for future in as_completed(self._futures):
#             self._futures.remove(future)
