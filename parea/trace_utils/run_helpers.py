"""Decorator for creating a run tree from functions."""
from typing import Any, Callable, Dict, List, Optional, TypedDict, Union

import contextvars
import inspect
import logging
import os
from collections.abc import Generator, Mapping
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from functools import wraps
from uuid import uuid4

from parea.client_two import ID_TYPE, Client
from parea.trace_utils.run_trees import RunTree

logger = logging.getLogger(__name__)
_PARENT_RUN_TREE = contextvars.ContextVar[Optional[RunTree]]("_PARENT_RUN_TREE", default=None)
_PROJECT_NAME = contextvars.ContextVar[Optional[str]]("_PROJECT_NAME", default=None)


def get_run_tree_context() -> Optional[RunTree]:
    """Get the current run tree context."""
    return _PARENT_RUN_TREE.get()


def _get_inputs(signature: inspect.Signature, *args: Any, **kwargs: Any) -> dict[str, Any]:
    """Return a dictionary of inputs from the function signature."""
    bound = signature.bind_partial(*args, **kwargs)
    bound.apply_defaults()
    arguments = dict(bound.arguments)
    arguments.pop("self", None)
    arguments.pop("cls", None)
    for param_name, param in signature.parameters.items():
        if param.kind == inspect.Parameter.VAR_KEYWORD:
            # Update with the **kwargs, and remove the original entry
            # This is to help flatten out keyword arguments
            if param_name in arguments:
                arguments.update(arguments[param_name])
                arguments.pop(param_name)

    return arguments


class PareaExtra(TypedDict):
    """Any additional info to be injected into the run dynamically."""

    reference_example_id: Optional[ID_TYPE]
    run_extra: Optional[dict]
    run_tree: Optional[RunTree]
    project_name: Optional[str]
    metadata: Optional[dict[str, Any]]
    tags: Optional[list[str]]
    run_id: Optional[ID_TYPE]


def traceable(
    run_type: str,
    *,
    name: Optional[str] = None,
    extra: Optional[dict] = None,
    executor: Optional[ThreadPoolExecutor] = None,
    metadata: Optional[Mapping[str, Any]] = None,
    tags: Optional[list[str]] = None,
) -> Callable:
    """Decorator for creating or adding a run to a run tree."""
    extra_outer = extra or {}

    def decorator(func: Callable):
        @wraps(func)
        async def async_wrapper(
            *args: Any,
            langsmith_extra: Optional[Union[PareaExtra, dict]] = None,
            **kwargs: Any,
        ) -> Any:
            """Async version of wrapper function"""
            outer_project = _PROJECT_NAME.get() or os.environ.get("PAREA_PROJECT", os.environ.get("PAREA_PROJECT", "default"))
            langsmith_extra = langsmith_extra or {}
            run_tree = langsmith_extra.get("run_tree", kwargs.get("run_tree", None))
            project_name_ = langsmith_extra.get("project_name", outer_project)
            run_extra = langsmith_extra.get("run_extra", None)
            reference_example_id = langsmith_extra.get("reference_example_id", None)
            if run_tree is None:
                parent_run_ = _PARENT_RUN_TREE.get()
            else:
                parent_run_ = run_tree
            signature = inspect.signature(func)
            name_ = name or func.__name__
            docstring = func.__doc__
            if run_extra:
                extra_inner = {**extra_outer, **run_extra}
            else:
                extra_inner = extra_outer
            metadata_ = {**(metadata or {}), **(langsmith_extra.get("metadata") or {})}
            if metadata_:
                extra_inner["metadata"] = metadata_
            inputs = _get_inputs(signature, *args, **kwargs)
            tags_ = (tags or []) + (langsmith_extra.get("tags") or [])
            id_ = langsmith_extra.get("run_id", uuid4())
            if parent_run_ is not None:
                new_run = parent_run_.create_child(
                    name=name_,
                    run_type=run_type,
                    serialized={
                        "name": name,
                        "signature": str(signature),
                        "doc": docstring,
                    },
                    inputs=inputs,
                    tags=tags_,
                    extra=extra_inner,
                    run_id=id_,
                )
            else:
                new_run = RunTree(
                    id=id_,
                    name=name_,
                    serialized={
                        "name": name,
                        "signature": str(signature),
                        "doc": docstring,
                    },
                    inputs=inputs,
                    run_type=run_type,
                    reference_example_id=reference_example_id,
                    project_name=project_name_,
                    extra=extra_inner,
                    tags=tags_,
                    executor=executor,
                    client=Client(api_key=os.getenv("DEV_API_KEY")),
                )
            new_run.post()
            _PROJECT_NAME.set(project_name_)
            _PARENT_RUN_TREE.set(new_run)
            func_accepts_parent_run = inspect.signature(func).parameters.get("run_tree", None) is not None
            try:
                if func_accepts_parent_run:
                    function_result = await func(*args, run_tree=new_run, **kwargs)
                else:
                    function_result = await func(*args, **kwargs)
            except Exception as e:
                new_run.end(error=str(e))
                new_run.patch()
                _PARENT_RUN_TREE.set(parent_run_)
                _PROJECT_NAME.set(outer_project)
                raise e
            _PARENT_RUN_TREE.set(parent_run_)
            _PROJECT_NAME.set(outer_project)
            if isinstance(function_result, dict):
                new_run.end(outputs=function_result)
            else:
                new_run.end(outputs={"output": function_result})
            new_run.patch()
            return function_result

        @wraps(func)
        def wrapper(
            *args: Any,
            langsmith_extra: Optional[Union[PareaExtra, dict]] = None,
            **kwargs: Any,
        ) -> Any:
            """Create a new run or create_child() if run is passed in kwargs."""
            outer_project = _PROJECT_NAME.get() or os.environ.get("PAREA_PROJECT", os.environ.get("PAREA_PROJECT", "default"))
            langsmith_extra = langsmith_extra or {}
            run_tree = langsmith_extra.get("run_tree", kwargs.get("run_tree", None))
            project_name_ = langsmith_extra.get("project_name", outer_project)
            run_extra = langsmith_extra.get("run_extra", None)
            reference_example_id = langsmith_extra.get("reference_example_id", None)
            if run_tree is None:
                parent_run_ = _PARENT_RUN_TREE.get()
            else:
                parent_run_ = run_tree
            signature = inspect.signature(func)
            name_ = name or func.__name__
            docstring = func.__doc__
            if run_extra:
                extra_inner = {**extra_outer, **run_extra}
            else:
                extra_inner = extra_outer
            metadata_ = {**(metadata or {}), **(langsmith_extra.get("metadata") or {})}
            if metadata_:
                extra_inner["metadata"] = metadata_
            inputs = _get_inputs(signature, *args, **kwargs)
            tags_ = (tags or []) + (langsmith_extra.get("tags") or [])
            id_ = langsmith_extra.get("run_id", uuid4())
            if parent_run_ is not None:
                new_run = parent_run_.create_child(
                    name=name_,
                    run_id=id_,
                    run_type=run_type,
                    serialized={
                        "name": name,
                        "signature": str(signature),
                        "doc": docstring,
                    },
                    inputs=inputs,
                    tags=tags_,
                    extra=extra_inner,
                )
            else:
                new_run = RunTree(
                    name=name_,
                    id=id_,
                    serialized={
                        "name": name,
                        "signature": str(signature),
                        "doc": docstring,
                    },
                    inputs=inputs,
                    run_type=run_type,
                    reference_example_id=reference_example_id,
                    project_name=project_name_,
                    extra=extra_inner,
                    tags=tags_,
                    executor=executor,
                )
            new_run.post()
            _PARENT_RUN_TREE.set(new_run)
            _PROJECT_NAME.set(project_name_)
            func_accepts_parent_run = inspect.signature(func).parameters.get("run_tree", None) is not None
            try:
                if func_accepts_parent_run:
                    function_result = func(*args, run_tree=new_run, **kwargs)
                else:
                    function_result = func(*args, **kwargs)
            except Exception as e:
                new_run.end(error=str(e))
                new_run.patch()
                _PARENT_RUN_TREE.set(parent_run_)
                _PROJECT_NAME.set(outer_project)
                raise e
            _PARENT_RUN_TREE.set(parent_run_)
            _PROJECT_NAME.set(outer_project)
            if isinstance(function_result, dict):
                new_run.end(outputs=function_result)
            else:
                new_run.end(outputs={"output": function_result})
            new_run.patch()
            return function_result

        if inspect.iscoroutinefunction(func):
            return async_wrapper
        else:
            return wrapper

    return decorator


@contextmanager
def trace(
    name: str,
    run_type: str,
    *,
    inputs: Optional[dict] = None,
    extra: Optional[dict] = None,
    executor: Optional[ThreadPoolExecutor] = None,
    project_name: Optional[str] = None,
    run_tree: Optional[RunTree] = None,
    tags: Optional[list[str]] = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> Generator[RunTree, None, None]:
    """Context manager for creating a run tree."""
    extra_outer = extra or {}
    if metadata:
        extra_outer["metadata"] = metadata
    parent_run_ = _PARENT_RUN_TREE.get() if run_tree is None else run_tree
    outer_project = _PROJECT_NAME.get() or os.environ.get("PAREA_PROJECT", os.environ.get("PAREA_PROJECT", "default"))
    project_name_ = project_name or outer_project
    if parent_run_ is not None:
        new_run = parent_run_.create_child(
            name=name,
            run_type=run_type,
            extra=extra_outer,
            inputs=inputs,
            tags=tags,
        )
    else:
        new_run = RunTree(
            name=name,
            run_type=run_type,
            extra=extra_outer,
            executor=executor,
            project_name=project_name_,
            inputs=inputs or {},
            tags=tags,
        )
    new_run.post()
    _PARENT_RUN_TREE.set(new_run)
    _PROJECT_NAME.set(project_name_)
    try:
        yield new_run
    except Exception as e:
        new_run.end(error=str(e))
        new_run.patch()
        _PARENT_RUN_TREE.set(parent_run_)
        _PROJECT_NAME.set(outer_project)
        raise e
    _PARENT_RUN_TREE.set(parent_run_)
    _PROJECT_NAME.set(outer_project)
    if new_run.end_time is None:
        # User didn't call end() on the run, so we'll do it for them
        new_run.end()
    new_run.patch()


# import contextvars
# import inspect
# import logging
# import os
# from concurrent.futures import ThreadPoolExecutor
# from contextlib import contextmanager
# from datetime import datetime
# from enum import Enum
# from functools import wraps
# from typing import Any, Callable, Dict, Generator, List, Mapping, Optional, Union, TypedDict
# from uuid import uuid4
#
# from parea.client_two import ID_TYPE, Client
# from parea.run_trees import RunTree
#
# logger = logging.getLogger(__name__)
# _PARENT_RUN_TREE = contextvars.ContextVar[Optional[RunTree]]("_PARENT_RUN_TREE", default=None)
# _PROJECT_NAME = contextvars.ContextVar[Optional[str]]("_PROJECT_NAME", default=None)
#
#
# def get_run_tree_context() -> Optional[RunTree]:
#     """Get the current run tree context."""
#     return _PARENT_RUN_TREE.get()
#
#
# def _get_inputs(signature: inspect.Signature, *args: Any, **kwargs: Any) -> Dict[str, Any]:
#     """Return a dictionary of inputs from the function signature."""
#     bound = signature.bind_partial(*args, **kwargs)
#     bound.apply_defaults()
#     arguments = dict(bound.arguments)
#     arguments.pop("self", None)
#     arguments.pop("cls", None)
#     for param_name, param in signature.parameters.items():
#         if param.kind == inspect.Parameter.VAR_KEYWORD:
#             # Update with the **kwargs, and remove the original entry
#             # This is to help flatten out keyword arguments
#             if param_name in arguments:
#                 arguments.update(arguments[param_name])
#                 arguments.pop(param_name)
#
#     return arguments
#
#
# class PareaExtra(TypedDict):
#     """Any additional info to be injected into the run dynamically."""
#
#     reference_example_id: Optional[ID_TYPE]
#     run_extra: Optional[Dict]
#     run_tree: Optional[RunTree]
#     project_name: Optional[str]
#     metadata: Optional[Dict[str, Any]]
#     tags: Optional[List[str]]
#     run_id: Optional[ID_TYPE]
#
#
# class RunTypeEnum(str, Enum):
#     """Enum for run types."""
#
#     tool = "tool"
#     chain = "chain"
#     llm = "llm"
#     retriever = "retriever"
#     embedding = "embedding"
#     prompt = "prompt"
#     parser = "parser"
#
#
# def _get_outer_project() -> Optional[str]:
#     return _PROJECT_NAME.get() or os.environ.get("PAREA_PROJECT", os.environ.get("PAREA_PROJECT", "default"))
#
#
# def _get_project_name(langsmith_extra: Optional[Union[PareaExtra, Dict]], outer_project: Optional[str]) -> Optional[str]:
#     return langsmith_extra.get("project_name", outer_project) if langsmith_extra else outer_project
#
#
# def _get_parent_run(run_tree: Optional[RunTree]) -> Optional[RunTree]:
#     return _PARENT_RUN_TREE.get() if run_tree is None else run_tree
#
#
# def _get_run_extra(langsmith_extra: Optional[Union[PareaExtra, Dict]], extra_outer: Optional[Dict]) -> Optional[Dict]:
#     run_extra = langsmith_extra.get("run_extra", None) if langsmith_extra else None
#     if run_extra:
#         return {**extra_outer, **run_extra}
#     else:
#         return extra_outer
#
#
# def _get_metadata(langsmith_extra: Optional[Union[PareaExtra, Dict]], metadata: Optional[Mapping[str, Any]]) -> Optional[Dict[str, Any]]:
#     metadata_ = {**(metadata or {}), **(langsmith_extra.get("metadata") or {})} if langsmith_extra else {}
#     if metadata_:
#         return metadata_
#     return None
#
#
# def _get_tags(langsmith_extra: Optional[Union[PareaExtra, Dict]], tags: Optional[List[str]]) -> Optional[List[str]]:
#     return (tags or []) + (langsmith_extra.get("tags") or []) if langsmith_extra else tags
#
#
# def _get_id(langsmith_extra: Optional[Union[PareaExtra, Dict]]) -> str:
#     return langsmith_extra.get("run_id", uuid4()) if langsmith_extra else uuid4()
#
#
# def _get_func_accepts_parent_run(func: Callable) -> bool:
#     return inspect.signature(func).parameters.get("run_tree", None) is not None
#
#
# async def _get_function_result(func: Callable, new_run: RunTree, func_accepts_parent_run: bool, *args: Any, **kwargs: Any) -> Any:
#     if func_accepts_parent_run:
#         return await func(*args, run_tree=new_run, **kwargs)
#     else:
#         return await func(*args, **kwargs)
#
#
# def _end_run(new_run: RunTree, function_result: Any) -> None:
#     if isinstance(function_result, dict):
#         new_run.end(outputs=function_result)
#     else:
#         new_run.end(outputs={"output": function_result})
#
#
# def create_run_tree(
#     run_type: RunTypeEnum,
#     name: str,
#     inputs: Dict[str, Any],
#     extra: Optional[Dict] = None,
#     tags: Optional[List[str]] = None,
#     id_: Optional[str] = None,
#     parent_run: Optional[RunTree] = None,
#     executor: Optional[ThreadPoolExecutor] = None,
#     reference_example_id: Optional[ID_TYPE] = None,
#     start_time: Optional[datetime] = None,
# ) -> RunTree:
#     """Helper function to create a new run tree or child run."""
#     if parent_run is not None:
#         return parent_run.create_child(
#             name=name,
#             run_type=run_type,
#             serialized={"name": name},
#             inputs=inputs,
#             tags=tags,
#             extra=extra,
#             run_id=id_,
#         )
#     else:
#         return RunTree(
#             name=name,
#             id=id_,
#             serialized={"name": name},
#             inputs=inputs,
#             run_type=run_type,
#             reference_example_id=reference_example_id,
#             start_time=start_time or datetime.utcnow(),
#             extra=extra,
#             tags=tags,
#             executor=executor,
#             client=Client(api_key=os.getenv("DEV_API_KEY")),
#         )
#
#
# def traceable(
#     run_type: RunTypeEnum,
#     *,
#     name: Optional[str] = None,
#     extra: Optional[Dict] = None,
#     executor: Optional[ThreadPoolExecutor] = None,
#     metadata: Optional[Mapping[str, Any]] = None,
#     tags: Optional[List[str]] = None,
# ) -> Callable:
#     """Decorator for creating or adding a run to a run tree."""
#     extra_outer = extra or {}
#
#     def decorator(func: Callable):
#         @wraps(func)
#         async def async_wrapper(
#             *args: Any,
#             langsmith_extra: Optional[Union[PareaExtra, Dict]] = None,
#             **kwargs: Any,
#         ) -> Any:
#             """Async version of wrapper function"""
#             outer_project = _get_outer_project()
#             project_name_ = _get_project_name(langsmith_extra, outer_project)
#             parent_run_ = _get_parent_run(langsmith_extra.get("run_tree", kwargs.get("run_tree", None)) if langsmith_extra else None)
#             signature = inspect.signature(func)
#             name_ = name or func.__name__
#             docstring = func.__doc__
#             extra_inner = _get_run_extra(langsmith_extra, extra_outer)
#             metadata_ = _get_metadata(langsmith_extra, metadata)
#             if metadata_:
#                 extra_inner["metadata"] = metadata_
#             inputs = _get_inputs(signature, *args, **kwargs)
#             tags_ = _get_tags(langsmith_extra, tags)
#             id_ = _get_id(langsmith_extra)
#             new_run = create_run_tree(run_type, name_, inputs, extra_inner, tags_, id_, parent_run_, project_name_, executor)
#             new_run.post()
#             _PROJECT_NAME.set(project_name_)
#             _PARENT_RUN_TREE.set(new_run)
#             func_accepts_parent_run = _get_func_accepts_parent_run(func)
#             try:
#                 function_result = await _get_function_result(func, new_run, func_accepts_parent_run, *args, **kwargs)
#             except Exception as e:
#                 new_run.end(error=str(e))
#                 new_run.patch()
#                 _PARENT_RUN_TREE.set(parent_run_)
#                 _PROJECT_NAME.set(outer_project)
#                 raise e
#             _PARENT_RUN_TREE.set(parent_run_)
#             _PROJECT_NAME.set(outer_project)
#             _end_run(new_run, function_result)
#             new_run.patch()
#             return function_result
#
#         return async_wrapper
#
#     return decorator
#
#
# @contextmanager
# def trace(
#     name: str,
#     run_type: RunTypeEnum,
#     *,
#     inputs: Optional[Dict] = None,
#     extra: Optional[Dict] = None,
#     executor: Optional[ThreadPoolExecutor] = None,
#     project_name: Optional[str] = None,
#     run_tree: Optional[RunTree] = None,
#     tags: Optional[List[str]] = None,
#     metadata: Optional[Mapping[str, Any]] = None,
# ) -> Generator[RunTree, None, None]:
#     """Context manager for creating a run tree."""
#     extra_outer = extra or {}
#     outer_project = _get_outer_project()
#     project_name_ = project_name or outer_project
#     parent_run_ = _get_parent_run(run_tree)
#     extra_inner = _get_run_extra(None, extra_outer)
#     metadata_ = _get_metadata(None, metadata)
#     if metadata_:
#         extra_inner["metadata"] = metadata_
#     new_run = create_run_tree(
#         run_type, name, inputs or {}, extra_inner, tags, str(uuid4()), parent_run_, project_name_, executor
#     )
#     new_run.post()
#     _PROJECT_NAME.set(project_name_)
#     _PARENT_RUN_TREE.set(new_run)
#     try:
#         yield new_run
#     except Exception as e:
#         new_run.end(error=str(e))
#         new_run.patch()
#         _PARENT_RUN_TREE.set(parent_run_)
#         _PROJECT_NAME.set(outer_project)
#         raise e
#     _PARENT_RUN_TREE.set(parent_run_)
#     _PROJECT_NAME.set(outer_project)
#     if new_run.end_time is None:
#         # User didn't call end() on the run, so we'll do it for them
#         new_run.end()
#     new_run.patch()
