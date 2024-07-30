from typing import Any, AsyncGenerator, AsyncIterator, Dict, Generator, Iterator

import contextvars
import inspect
import json
import logging
import os
from datetime import datetime

from openai import AsyncOpenAI
from openai.types.beta.threads import Run

from parea.constants import PAREA_OS_ENV_EXPERIMENT_UUID
from parea.helpers import gen_trace_id, is_logging_disabled, timezone_aware_now
from parea.schemas import TraceLog, UpdateTraceScenario
from parea.utils.trace_utils import check_multiple_return_values, logger_all_possible, make_output, trace_context, trace_data
from parea.utils.universal_encoder import json_dumps
from parea.wrapper.utils import _compute_cost

logger = logging.getLogger()


class BaseWrapper:
    @staticmethod
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
            elif scenario == UpdateTraceScenario.USAGE:
                if usage := data.get("usage", {}):
                    if usage:
                        trace_data.get()[trace_id].input_tokens = usage.get("prompt_tokens", 0)
                        trace_data.get()[trace_id].output_tokens = usage.get("completion_tokens", 0)
                        trace_data.get()[trace_id].total_tokens = usage.get("total_tokens", 0)
                        if data.get("model", None):
                            trace_data.get()[trace_id].cost = _compute_cost(usage.get("prompt_tokens", 0), usage.get("total_tokens", 0), data.get("model", None))
            else:
                logger.debug(f"Error occurred filling trace data. Scenario not valid: {scenario}")
        except Exception as e:
            logger.debug(
                f"Error occurred filling trace data for trace id {trace_id}, {e}",
                exc_info=e,
            )

    def init_trace(self, func_name, args, kwargs, func):
        start_time = timezone_aware_now()
        trace_id = gen_trace_id()

        new_trace_context = trace_context.get() + [trace_id]
        token = trace_context.set(new_trace_context)

        if is_logging_disabled():
            return trace_id, start_time, token

        try:
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
                    except TypeError:
                        # if we can't serialize the value, just convert it to a string
                        inputs[k] = str(v)

            trace_data.get()[trace_id] = TraceLog(
                trace_id=trace_id,
                parent_trace_id=trace_id,
                root_trace_id=new_trace_context[0],
                start_timestamp=start_time.isoformat(),
                trace_name=func_name,
                inputs=inputs,
                experiment_uuid=os.environ.get(PAREA_OS_ENV_EXPERIMENT_UUID, None),
            )

            parent_trace_id = new_trace_context[-2] if len(new_trace_context) > 1 else None
            if parent_trace_id:
                self.fill_trace_data(
                    trace_id,
                    {"parent_trace_id": parent_trace_id},
                    UpdateTraceScenario.CHAIN,
                )
        except Exception as e:
            logger.debug(f"Error occurred initializing trace for function {func_name}, {e}")

        return trace_id, start_time, token

    @staticmethod
    def cleanup_trace(trace_id: str, start_time: datetime, context_token: contextvars.Token):
        end_time = timezone_aware_now()
        trace_data.get()[trace_id].end_timestamp = end_time.isoformat()
        trace_data.get()[trace_id].latency = (end_time - start_time).total_seconds()

        output = trace_data.get()[trace_id].output
        if trace_data.get()[trace_id].status == "success" and output:
            output_for_eval_metrics = output
            trace_data.get()[trace_id].output_for_eval_metrics = output_for_eval_metrics

        logger_all_possible(trace_id)
        trace_context.reset(context_token)

    def _wrap_steps(self, module_name, func, *args, **kwargs):
        trace_id, start_time, context_token = self.init_trace(f"{module_name}.{func.__name__}", args, kwargs, func)
        output_as_list = check_multiple_return_values(func)
        try:
            result = func(*args, **kwargs)
            self.fill_trace_data(
                trace_id,
                {
                    "result": result,
                    "output_as_list": output_as_list,
                },
                UpdateTraceScenario.RESULT,
            )
            if isinstance(result, Run) and result.usage:
                self.fill_trace_data(
                    trace_id,
                    result.model_dump(),
                    UpdateTraceScenario.USAGE,
                )
            return result
        except Exception as e:
            logger.error(f"Error occurred in function {func.__name__}, {e}")
            self.fill_trace_data(trace_id, {"error": str(e)}, UpdateTraceScenario.ERROR)
            raise e
        finally:
            try:
                self.cleanup_trace(trace_id, start_time, context_token)
            except Exception as e:
                logger.debug(
                    f"Error occurred cleaning up trace for function {func.__name__}, {e}",
                    exc_info=e,
                )

    async def _awrap_steps(self, module_name, func, *args, **kwargs):
        trace_id, start_time, context_token = self.init_trace(f"{module_name}.{func.__name__}", args, kwargs, func)
        output_as_list = check_multiple_return_values(func)
        try:
            result = await func(*args, **kwargs)
            self.fill_trace_data(
                trace_id,
                {
                    "result": result,
                    "output_as_list": output_as_list,
                },
                UpdateTraceScenario.RESULT,
            )
            if isinstance(result, Run) and result.usage:
                self.fill_trace_data(
                    trace_id,
                    result.model_dump(),
                    UpdateTraceScenario.USAGE,
                )
            return result
        except Exception as e:
            logger.error(f"Error occurred in function {func.__name__}, {e}")
            self.fill_trace_data(trace_id, {"error": str(e)}, UpdateTraceScenario.ERROR)
            raise e
        finally:
            try:
                self.cleanup_trace(trace_id, start_time, context_token)
            except Exception as e:
                logger.debug(
                    f"Error occurred cleaning up trace for function {func.__name__}, {e}",
                    exc_info=e,
                )


class AssistantsWrapper(BaseWrapper):
    def __init__(self, client, is_async):
        self.module_name = "beta.assistants"
        self.client = client
        self.wrapped = False
        self.is_async = is_async
        try:
            self.create_method = client.beta.assistants.create
            self.retrieve_method = client.beta.assistants.retrieve
            self.update_method = client.beta.assistants.update
            self.list_method = client.beta.assistants.list
            self.delete_method = client.beta.assistants.delete
            self.wrapped = True
        except Exception as e:
            logger.debug(f"Error occurred initializing AssistantsWrapper, {e}")

    def create(self, *args, **kwargs):
        func = self.create_method
        return self._wrap_steps(self.module_name, func, *args, **kwargs)

    def retrieve(self, *args, **kwargs):
        func = self.retrieve_method
        return self._wrap_steps(self.module_name, func, *args, **kwargs)

    def update(self, *args, **kwargs):
        func = self.update_method
        return self._wrap_steps(self.module_name, func, *args, **kwargs)

    def list(self, *args, **kwargs):
        func = self.list_method
        return self._wrap_steps(self.module_name, func, *args, **kwargs)

    def delete(self, *args, **kwargs):
        func = self.delete_method
        return self._wrap_steps(self.module_name, func, *args, **kwargs)

    async def acreate(self, *args, **kwargs):
        func = self.create_method
        return await self._awrap_steps(self.module_name, func, *args, **kwargs)

    async def aretrieve(self, *args, **kwargs):
        func = self.retrieve_method
        return await self._awrap_steps(self.module_name, func, *args, **kwargs)

    async def aupdate(self, *args, **kwargs):
        func = self.update_method
        return await self._awrap_steps(self.module_name, func, *args, **kwargs)

    async def alist(self, *args, **kwargs):
        func = self.list_method
        return await self._awrap_steps(self.module_name, func, *args, **kwargs)

    async def adelete(self, *args, **kwargs):
        func = self.delete_method
        return await self._awrap_steps(self.module_name, func, *args, **kwargs)

    def init(self):
        if not self.wrapped:
            return

        methods = {
            self.create_method: self.acreate if self.is_async else self.create,
            self.retrieve_method: self.aretrieve if self.is_async else self.retrieve,
            self.update_method: self.aupdate if self.is_async else self.update,
            self.list_method: self.alist if self.is_async else self.list,
            self.delete_method: self.adelete if self.is_async else self.delete,
        }

        for method, function in methods.items():
            setattr(self.client.beta.assistants, method.__name__, function)


class ThreadsWrapper(BaseWrapper):
    def __init__(self, client, is_async):
        self.module_name = "beta.threads"
        self.client = client
        self.wrapped = False
        self.is_async = is_async
        try:
            self.create_method = client.beta.threads.create
            self.retrieve_method = client.beta.threads.retrieve
            self.update_method = client.beta.threads.update
            self.create_and_run_method = client.beta.threads.create_and_run
            self.delete_method = client.beta.threads.delete
            self.wrapped = True
        except Exception as e:
            logger.debug(f"Error occurred initializing ThreadsWrapper, {e}")

    def create(self, *args, **kwargs):
        func = self.create_method
        return self._wrap_steps(self.module_name, func, *args, **kwargs)

    def retrieve(self, *args, **kwargs):
        func = self.retrieve_method
        return self._wrap_steps(self.module_name, func, *args, **kwargs)

    def update(self, *args, **kwargs):
        func = self.update_method
        return self._wrap_steps(self.module_name, func, *args, **kwargs)

    def create_and_run(self, *args, **kwargs):
        if kwargs.get("stream", False):
            logger.debug("Stream is not supported for create_and_run method")
            return None
        func = self.create_and_run_method
        return self._wrap_steps(self.module_name, func, *args, **kwargs)

    def delete(self, *args, **kwargs):
        func = self.delete_method
        return self._wrap_steps(self.module_name, func, *args, **kwargs)

    async def acreate(self, *args, **kwargs):
        func = self.create_method
        return await self._awrap_steps(self.module_name, func, *args, **kwargs)

    async def aretrieve(self, *args, **kwargs):
        func = self.retrieve_method
        return await self._awrap_steps(self.module_name, func, *args, **kwargs)

    async def aupdate(self, *args, **kwargs):
        func = self.update_method
        return await self._awrap_steps(self.module_name, func, *args, **kwargs)

    async def acreate_and_run(self, *args, **kwargs):
        if kwargs.get("stream", False):
            logger.debug("Stream is not supported for create_and_run method")
            return None
        func = self.create_and_run_method
        return self._awrap_steps(self.module_name, func, *args, **kwargs)

    async def adelete(self, *args, **kwargs):
        func = self.delete_method
        return await self._awrap_steps(self.module_name, func, *args, **kwargs)

    def init(self):
        if not self.wrapped:
            return

        methods = {
            self.create_method: self.acreate if self.is_async else self.create,
            self.retrieve_method: self.aretrieve if self.is_async else self.retrieve,
            self.update_method: self.aupdate if self.is_async else self.update,
            self.create_and_run_method: self.acreate_and_run if self.is_async else self.create_and_run,
            self.delete_method: self.adelete if self.is_async else self.delete,
        }

        for method, function in methods.items():
            setattr(self.client.beta.threads, method.__name__, function)


class ThreadRunsWrapper(BaseWrapper):
    def __init__(self, client, is_async):
        self.module_name = "beta.threads.runs"
        self.client = client
        self.wrapped = False
        self.is_async = is_async
        try:
            self.create_method = client.beta.threads.runs.create
            self.retrieve_method = client.beta.threads.runs.retrieve
            self.update_method = client.beta.threads.runs.update
            self.list_method = client.beta.threads.runs.list
            self.submit_tool_outputs_method = client.beta.threads.runs.submit_tool_outputs
            self.cancel_method = client.beta.threads.runs.cancel
            self.wrapped = True
        except Exception as e:
            logger.debug(f"Error occurred initializing ThreadRunsWrapper, {e}")

    def create(self, *args, **kwargs):
        if kwargs.get("stream", False):
            logger.debug("Stream is not supported for create method")
            return None
        func = self.create_method
        return self._wrap_steps(self.module_name, func, *args, **kwargs)

    def retrieve(self, *args, **kwargs):
        func = self.retrieve_method
        return self._wrap_steps(self.module_name, func, *args, **kwargs)

    def update(self, *args, **kwargs):
        func = self.update_method
        return self._wrap_steps(self.module_name, func, *args, **kwargs)

    def list(self, *args, **kwargs):
        func = self.list_method
        return self._wrap_steps(self.module_name, func, *args, **kwargs)

    def cancel(self, *args, **kwargs):
        func = self.cancel_method
        return self._wrap_steps(self.module_name, func, *args, **kwargs)

    def submit_tool_outputs(self, *args, **kwargs):
        if kwargs.get("stream", False):
            logger.debug("Stream is not supported for submit_tool_outputs method")
            return None
        func = self.submit_tool_outputs_method
        return self._wrap_steps(self.module_name, func, *args, **kwargs)

    async def acreate(self, *args, **kwargs):
        if kwargs.get("stream", False):
            logger.debug("Stream is not supported for create method")
            return None
        func = self.create_method
        return await self._awrap_steps(self.module_name, func, *args, **kwargs)

    async def aretrieve(self, *args, **kwargs):
        func = self.retrieve_method
        return await self._awrap_steps(self.module_name, func, *args, **kwargs)

    async def aupdate(self, *args, **kwargs):
        func = self.update_method
        return await self._awrap_steps(self.module_name, func, *args, **kwargs)

    async def alist(self, *args, **kwargs):
        func = self.list_method
        return await self._awrap_steps(self.module_name, func, *args, **kwargs)

    async def acancel(self, *args, **kwargs):
        func = self.cancel_method
        return await self._awrap_steps(self.module_name, func, *args, **kwargs)

    async def asubmit_tool_outputs(self, *args, **kwargs):
        if kwargs.get("stream", False):
            logger.debug("Stream is not supported for submit_tool_outputs method")
            return None
        func = self.submit_tool_outputs_method
        return await self._awrap_steps(self.module_name, func, *args, **kwargs)

    def init(self):
        if not self.wrapped:
            return

        methods = {
            self.create_method: self.acreate if self.is_async else self.create,
            self.retrieve_method: self.aretrieve if self.is_async else self.retrieve,
            self.update_method: self.aupdate if self.is_async else self.update,
            self.list_method: self.alist if self.is_async else self.list,
            self.cancel_method: self.acancel if self.is_async else self.cancel,
            self.submit_tool_outputs_method: self.asubmit_tool_outputs if self.is_async else self.submit_tool_outputs,
        }

        for method, function in methods.items():
            setattr(self.client.beta.threads.runs, method.__name__, function)


class ThreadMessagesWrapper(BaseWrapper):
    def __init__(self, client, is_async):
        self.module_name = "beta.threads.messages"
        self.client = client
        self.wrapped = False
        self.is_async = is_async
        try:
            self.create_method = client.beta.threads.messages.create
            self.retrieve_method = client.beta.threads.messages.retrieve
            self.update_method = client.beta.threads.messages.update
            self.list_method = client.beta.threads.messages.list
            self.wrapped = True
        except Exception as e:
            logger.debug(f"Error occurred initializing ThreadMessagesWrapper, {e}")

    def create(self, *args, **kwargs):
        func = self.create_method
        return self._wrap_steps(self.module_name, func, *args, **kwargs)

    def retrieve(self, *args, **kwargs):
        func = self.retrieve_method
        return self._wrap_steps(self.module_name, func, *args, **kwargs)

    def update(self, *args, **kwargs):
        func = self.update_method
        return self._wrap_steps(self.module_name, func, *args, **kwargs)

    def list(self, *args, **kwargs):
        func = self.list_method
        return self._wrap_steps(self.module_name, func, *args, **kwargs)

    async def acreate(self, *args, **kwargs):
        func = self.create_method
        return await self._awrap_steps(self.module_name, func, *args, **kwargs)

    async def aretrieve(self, *args, **kwargs):
        func = self.retrieve_method
        return await self._awrap_steps(self.module_name, func, *args, **kwargs)

    async def aupdate(self, *args, **kwargs):
        func = self.update_method
        return await self._awrap_steps(self.module_name, func, *args, **kwargs)

    async def alist(self, *args, **kwargs):
        func = self.list_method
        return await self._awrap_steps(self.module_name, func, *args, **kwargs)

    def init(self):
        if not self.wrapped:
            return

        methods = {
            self.create_method: self.acreate if self.is_async else self.create,
            self.retrieve_method: self.aretrieve if self.is_async else self.retrieve,
            self.update_method: self.aupdate if self.is_async else self.update,
            self.list_method: self.alist if self.is_async else self.list,
        }

        for method, function in methods.items():
            setattr(self.client.beta.threads.messages, method.__name__, function)


class ThreadRunsStepsWrapper(BaseWrapper):
    def __init__(self, client, is_async):
        self.module_name = "beta.threads.runs.steps"
        self.client = client
        self.wrapped = False
        self.is_async = is_async
        try:
            self.retrieve_method = client.beta.threads.runs.steps.retrieve
            self.list_method = client.beta.threads.runs.steps.list
            self.wrapped = True
        except Exception as e:
            logger.debug(f"Error occurred initializing ThreadRunsStepsWrapper, {e}")

    def retrieve(self, *args, **kwargs):
        func = self.retrieve_method
        return self._wrap_steps(self.module_name, func, *args, **kwargs)

    def list(self, *args, **kwargs):
        func = self.list_method
        return self._wrap_steps(self.module_name, func, *args, **kwargs)

    async def aretrieve(self, *args, **kwargs):
        func = self.retrieve_method
        return await self._awrap_steps(self.module_name, func, *args, **kwargs)

    async def alist(self, *args, **kwargs):
        func = self.list_method
        return await self._awrap_steps(self.module_name, func, *args, **kwargs)

    def init(self):
        if not self.wrapped:
            return

        methods = {
            self.retrieve_method: self.aretrieve if self.is_async else self.retrieve,
            self.list_method: self.alist if self.is_async else self.list,
        }

        for method, function in methods.items():
            setattr(self.client.beta.threads.runs.steps, method.__name__, function)


class ThreadMessagesFilesWrapper(BaseWrapper):
    def __init__(self, client, is_async):
        self.module_name = "beta.threads.messages.files"
        self.client = client
        self.wrapped = False
        self.is_async = is_async
        try:
            self.retrieve_method = client.beta.threads.messages.files.retrieve
            self.list_method = client.beta.threads.messages.files.list
            self.wrapped = True
        except Exception as e:
            logger.debug(f"Error occurred initializing ThreadMessagesFilesWrapper, {e}")

    def retrieve(self, *args, **kwargs):
        func = self.retrieve_method
        return self._wrap_steps(self.module_name, func, *args, **kwargs)

    def list(self, *args, **kwargs):
        func = self.list_method
        return self._wrap_steps(self.module_name, func, *args, **kwargs)

    async def aretrieve(self, *args, **kwargs):
        func = self.retrieve_method
        return await self._awrap_steps(self.module_name, func, *args, **kwargs)

    async def alist(self, *args, **kwargs):
        func = self.list_method
        return await self._awrap_steps(self.module_name, func, *args, **kwargs)

    def init(self):
        if not self.wrapped:
            return

        methods = {
            self.retrieve_method: self.aretrieve if self.is_async else self.retrieve,
            self.list_method: self.alist if self.is_async else self.list,
        }

        for method, function in methods.items():
            setattr(self.client.beta.threads.messages.files, method.__name__, function)


class BetaWrappers:
    def __init__(self, client):
        self.is_async = isinstance(client, AsyncOpenAI)
        self.client = client

    def init(self):
        aw = AssistantsWrapper(self.client, self.is_async)
        tw = ThreadsWrapper(self.client, self.is_async)
        trw = ThreadRunsWrapper(self.client, self.is_async)
        trsw = ThreadRunsStepsWrapper(self.client, self.is_async)
        tmw = ThreadMessagesWrapper(self.client, self.is_async)
        tmfw = ThreadMessagesFilesWrapper(self.client, self.is_async)
        aw.init()
        tw.init()
        trw.init()
        trsw.init()
        tmw.init()
        tmfw.init()
