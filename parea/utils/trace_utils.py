from typing import Any, AsyncGenerator, AsyncIterator, Callable, Dict, Generator, Iterator, List, Optional, Tuple

import contextvars
import inspect
import json
import logging
import os
import threading
import traceback
from collections import ChainMap
from datetime import datetime
from functools import wraps
from random import random

from parea.constants import PAREA_OS_ENV_EXPERIMENT_UUID, TURN_OFF_PAREA_EVAL_LOGGING
from parea.helpers import gen_trace_id, is_logging_disabled, timezone_aware_now
from parea.parea_logger import parea_logger
from parea.schemas import EvaluationResult
from parea.schemas.models import TraceLog, UpdateLog, UpdateTraceScenario
from parea.utils.universal_encoder import json_dumps

logger = logging.getLogger()


# Context variable to maintain the current trace context stack
trace_context = contextvars.ContextVar("trace_context", default=[])

# A dictionary to hold trace data for each trace
trace_data = contextvars.ContextVar("trace_data", default={})

# Context variable to maintain running evals in thread
thread_ids_running_evals = contextvars.ContextVar("thread_ids_running_evals", default=[])

# Add a counter variable to maintain the execution order
execution_order_counters = contextvars.ContextVar("execution_order_counters", default={})


def clear_trace_context():
    global trace_context, trace_data, thread_ids_running_evals, execution_order_counters
    trace_context = contextvars.ContextVar("trace_context", default=[])
    trace_data = contextvars.ContextVar("trace_data", default={})
    thread_ids_running_evals = contextvars.ContextVar("thread_ids_running_evals", default=[])
    execution_order_counters = contextvars.ContextVar("execution_order_counters", default={})


def log_in_thread(target_func: Callable, data: Dict[str, Any]):
    context = contextvars.copy_context()
    logging_thread = threading.Thread(target=context.run, args=(target_func,), kwargs=data)
    logging_thread.start()


def merge(old, new):
    if isinstance(old, dict) and isinstance(new, dict):
        return dict(ChainMap(new, old))
    if isinstance(old, list) and isinstance(new, list):
        return old + new
    return new


def check_multiple_return_values(func) -> bool:
    specs = inspect.getfullargspec(func)
    try:
        r = specs.annotations.get("return")
        if r and r.__origin__ == tuple:
            return len(r.__args__) > 1
    except Exception:
        return False


def make_output(result, islist) -> Optional[str]:
    if result is None:
        return result
    try:
        if islist:
            json_list = [json_dumps(r) for r in result]
            return json_dumps(json_list)
        elif isinstance(result, str):
            return result
        else:
            return json_dumps(result)
    except Exception as e:
        logger.debug(f"Error occurred making output with result: {result}. Error: {e}", exc_info=e)
        return str(result)


def get_current_trace_id() -> str:
    stack = trace_context.get()
    if stack:
        return stack[-1]
    return ""


def get_root_trace_id() -> str:
    stack = trace_context.get()
    if stack:
        return stack[0]
    return ""


def trace_insert(data: Dict[str, Any], trace_id: Optional[str] = None):
    """
    Insert data into the trace log for the current or specified trace id. Data should be a dictionary with keys that correspond to the fields of the TraceLog model.
    If the field already has an existing value that is extensible (dict, set, list, etc.), the new value will be merged with the existing value.
    Args:
        data: Keys can be one of: trace_name, end_user_identifier, metadata, tags, deployment_id, images, session_id
        trace_id: The trace id to insert the data into. If not provided, the current trace id will be used.
    """
    try:
        current_trace_id = trace_id or get_current_trace_id()
        current_trace_data: TraceLog = trace_data.get()[current_trace_id]
        if not current_trace_data:
            return
        for key, new_value in data.items():
            existing_value = current_trace_data.__getattribute__(key)
            current_trace_data.__setattr__(key, merge(existing_value, new_value) if existing_value else new_value)
    except Exception as e:
        logger.debug(f"Error occurred inserting data into trace log, {e}", exc_info=e)


def fill_trace_data(trace_id: str, data: Dict[str, Any], scenario: UpdateTraceScenario):
    try:
        if scenario == UpdateTraceScenario.RESULT:
            if not isinstance(data["result"], (Generator, AsyncGenerator, AsyncIterator, Iterator)):
                trace_data.get()[trace_id].output = make_output(data["result"], data.get("output_as_list", False))
            trace_data.get()[trace_id].status = "success"
            trace_data.get()[trace_id].evaluation_metric_names = data.get("eval_funcs_names")
        elif scenario == UpdateTraceScenario.ERROR:
            trace_data.get()[trace_id].error = data["error"]
            trace_data.get()[trace_id].status = "error"
        elif scenario == UpdateTraceScenario.CHAIN:
            trace_data.get()[trace_id].parent_trace_id = data["parent_trace_id"]
            trace_data.get()[data["parent_trace_id"]].children.append(trace_id)
        elif scenario == UpdateTraceScenario.LANGCHAIN_CHILD:
            trace_data.get()[data["parent_trace_id"]].children.append(trace_id)
        elif scenario == UpdateTraceScenario.OPENAICONFIG:
            trace_data.get()[trace_id].configuration = data["configuration"]
            trace_data.get()[trace_id].output = data["output"]
            trace_data.get()[trace_id].input_tokens = data["input_tokens"]
            trace_data.get()[trace_id].output_tokens = data["output_tokens"]
            trace_data.get()[trace_id].total_tokens = data["total_tokens"]
            trace_data.get()[trace_id].cost = data["cost"]
        else:
            logger.debug(f"Error occurred filling trace data. Scenario not valid: {scenario}")
    except Exception as e:
        logger.debug(f"Error occurred filling trace data for trace id {trace_id}, {e}", exc_info=e)


def trace(
    name: Optional[str] = None,
    tags: Optional[List[str]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    end_user_identifier: Optional[str] = None,
    session_id: Optional[str] = None,
    eval_funcs_names: Optional[List[str]] = None,
    eval_funcs: Optional[List[Callable]] = None,
    access_output_of_func: Optional[Callable] = None,
    apply_eval_frac: float = 1.0,
    deployment_id: Optional[str] = None,
    log_omit_inputs: Optional[bool] = False,
    log_omit_outputs: Optional[bool] = False,
    overwrite_trace_id: Optional[str] = None,
    overwrite_inputs: Optional[Dict[str, Any]] = None,
):
    def init_trace(func_name, _parea_target_field, args, kwargs, func) -> Tuple[str, datetime, contextvars.Token]:
        start_time = timezone_aware_now()
        trace_id = overwrite_trace_id or gen_trace_id()

        new_trace_context = trace_context.get() + [trace_id]
        token = trace_context.set(new_trace_context)

        if is_logging_disabled():
            return trace_id, start_time, token

        try:
            if overwrite_inputs is not None:
                inputs = overwrite_inputs
            else:
                sig = inspect.signature(func)
                parameters = sig.parameters

                inputs = {k: v for k, v in zip(parameters.keys(), args)}
                inputs.update(kwargs)

            # filter out any values which aren't JSON serializable
            for k, v in inputs.items():
                try:
                    json.dumps(v)
                except TypeError:
                    try:
                        inputs[k] = json_dumps(v)
                    except (TypeError, AttributeError):
                        # if we can't serialize the value, just convert it to a string
                        inputs[k] = str(v)

            depth = len(new_trace_context) - 1
            root_trace_id = new_trace_context[0]

            # Get the execution order counter for the current root trace
            counters = execution_order_counters.get()
            if root_trace_id not in counters:
                counters[root_trace_id] = 0
            execution_order = counters[root_trace_id]
            counters[root_trace_id] += 1

            parent_trace_id = new_trace_context[-2] if len(new_trace_context) > 1 else None
            parent_target = trace_data.get()[parent_trace_id].target if parent_trace_id else None

            trace_data.get()[trace_id] = TraceLog(
                trace_id=trace_id,
                parent_trace_id=trace_id,
                root_trace_id=root_trace_id,
                start_timestamp=start_time.isoformat(),
                trace_name=name or func_name,
                end_user_identifier=end_user_identifier,
                session_id=session_id,
                metadata=metadata,
                target=_parea_target_field if _parea_target_field is not None else parent_target,
                tags=tags,
                inputs={} if log_omit_inputs else inputs,
                experiment_uuid=os.environ.get(PAREA_OS_ENV_EXPERIMENT_UUID, None),
                apply_eval_frac=apply_eval_frac,
                deployment_id=deployment_id,
                depth=depth,
                execution_order=execution_order,
            )

            if parent_trace_id:
                fill_trace_data(trace_id, {"parent_trace_id": parent_trace_id}, UpdateTraceScenario.CHAIN)
        except Exception as e:
            logger.debug(f"Error occurred initializing trace for function {func_name}, {e}")

        return trace_id, start_time, token

    def cleanup_trace(trace_id: str, start_time: datetime, context_token: contextvars.Token):
        end_time = timezone_aware_now()
        trace_data.get()[trace_id].end_timestamp = end_time.isoformat()
        trace_data.get()[trace_id].latency = (end_time - start_time).total_seconds()

        output = trace_data.get()[trace_id].output
        if trace_data.get()[trace_id].status == "success" and output:
            output_for_eval_metrics = output
            if access_output_of_func:
                try:
                    output = json.loads(output)
                    output = access_output_of_func(output)
                    output_for_eval_metrics = json_dumps(output)
                except Exception as e:
                    logger.error(f"Error accessing output of func with output: {output}. Error: {e}", exc_info=e)
            trace_data.get()[trace_id].output_for_eval_metrics = output_for_eval_metrics

        thread_eval_funcs_then_log(trace_id, eval_funcs)
        trace_context.reset(context_token)

    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            _parea_target_field = kwargs.pop("_parea_target_field", None)
            trace_id, start_time, context_token = init_trace(func.__name__, _parea_target_field, args, kwargs, func)
            output_as_list = check_multiple_return_values(func)
            try:
                result = await func(*args, **kwargs)
                if not is_logging_disabled() and not log_omit_outputs:
                    fill_trace_data(trace_id, {"result": result, "output_as_list": output_as_list, "eval_funcs_names": eval_funcs_names}, UpdateTraceScenario.RESULT)
                return result
            except Exception as e:
                logger.error(f"Error occurred in function {func.__name__}, {e}")
                fill_trace_data(trace_id, {"error": traceback.format_exc()}, UpdateTraceScenario.ERROR)
                raise e
            finally:
                try:
                    cleanup_trace(trace_id, start_time, context_token)
                except Exception as e:
                    logger.debug(f"Error occurred cleaning up trace for function {func.__name__}, {e}", exc_info=e)

        @wraps(func)
        def wrapper(*args, **kwargs):
            _parea_target_field = kwargs.pop("_parea_target_field", None)
            trace_id, start_time, context_token = init_trace(func.__name__, _parea_target_field, args, kwargs, func)
            output_as_list = check_multiple_return_values(func)
            try:
                result = func(*args, **kwargs)
                if not is_logging_disabled() and not log_omit_outputs:
                    fill_trace_data(trace_id, {"result": result, "output_as_list": output_as_list, "eval_funcs_names": eval_funcs_names}, UpdateTraceScenario.RESULT)
                return result
            except Exception as e:
                logger.error(f"Error occurred in function {func.__name__}, {e}")
                fill_trace_data(trace_id, {"error": traceback.format_exc()}, UpdateTraceScenario.ERROR)
                raise e
            finally:
                try:
                    cleanup_trace(trace_id, start_time, context_token)
                except Exception as e:
                    logger.debug(f"Error occurred cleaning up trace for function {func.__name__}, {e}", exc_info=e)

        if inspect.iscoroutinefunction(func):
            return async_wrapper
        else:
            return wrapper

    if callable(name):
        func = name
        name = None
        return decorator(func)

    return decorator


def call_eval_funcs_then_log(trace_id: str, eval_funcs: List[Callable] = None):
    data = trace_data.get()[trace_id]

    if eval_funcs and data.status == "success" and random() <= data.apply_eval_frac:
        thread_ids_running_evals.get().append(trace_id)
        if data.target is None and (root_target := trace_data.get()[data.root_trace_id].target) is not None:
            data.target = root_target
        scores = []
        for func in eval_funcs:
            try:
                if TURN_OFF_PAREA_EVAL_LOGGING:
                    score = func(data)
                else:
                    score = trace()(func)(data)
                if isinstance(score, EvaluationResult):
                    scores.append(score)
                elif isinstance(score, list):
                    scores.extend(score)
                elif score is not None:
                    scores.append(EvaluationResult(name=func.__name__, score=score))
            except Exception as e:
                logger.error(f"Error occurred calling evaluation function '{func.__name__}', {e}", exc_info=e)
        trace_data.get()[trace_id].scores = scores
        thread_ids_running_evals.get().remove(trace_id)

    data_with_scores = trace_data.get()[trace_id]
    parea_logger.default_log(data=data_with_scores)


def logger_record_log(trace_id: str):
    log_in_thread(parea_logger.record_log, {"data": trace_data.get()[trace_id]})


def logger_all_possible(trace_id: str):
    log_in_thread(parea_logger.default_log, {"data": trace_data.get()[trace_id]})


def thread_eval_funcs_then_log(trace_id: str, eval_funcs: List[Callable] = None):
    log_in_thread(call_eval_funcs_then_log, {"trace_id": trace_id, "eval_funcs": eval_funcs})


def logger_update_record(data: UpdateLog):
    log_in_thread(parea_logger.update_log, {"data": data})
