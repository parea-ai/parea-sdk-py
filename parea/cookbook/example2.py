from typing import Dict, List

import os

from dotenv import load_dotenv

from parea.client_two import Client
from parea.schemas.models import Completion, LLMInputs, Message, ModelParams, Role
from parea.trace_utils.run_helpers import traceable

load_dotenv()

client = Client(api_key=os.getenv("DEV_API_KEY"))

LIMIT = 1


# dump task array to string
def dump_task(task):
    d = ""  # init
    for tasklet in task:
        d += f"\n{tasklet.get('task_name','')}"
    d = d.strip()
    return d


@traceable(run_type="llm")
def open_ai_inference(
    prompt: str,
    model: str = "gpt-3.5-turbo",
    temperature: float = 0.5,
) -> str:
    return client.completion(
        data=Completion(
            llm_configuration=LLMInputs(
                model=model,
                provider="openai",
                model_params=ModelParams(model=model, temp=temperature),
                messages=[Message(role=Role.system, content=prompt)],
            ),
        )
    ).content.strip()


@traceable(run_type="chain")
def expound_task(main_objective: str, current_task: str) -> list[dict[str, str]]:
    print(f"****Expounding based on task:**** {current_task}")
    prompt = f"You are an AI who performs one task based on the following objective: {main_objective}\n" f"Your task: {current_task}\nResponse:"
    response = open_ai_inference(prompt)
    new_tasks = response.split("\n") if "\n" in response else [response]
    return [{"task_name": task_name} for task_name in new_tasks][0:1]


@traceable(run_type="chain")
def generate_tasks(main_objective: str, expounded_initial_task: list[dict[str, str]]) -> list[str]:
    task_expansion = dump_task(expounded_initial_task)
    prompt = (
        f"You are an AI who creates tasks based on the following MAIN OBJECTIVE: {main_objective}\n"
        f"Create tasks pertaining directly to your previous research here:\n"
        f"{task_expansion}\nResponse:"
    )
    response = open_ai_inference(prompt)
    new_tasks = response.split("\n") if "\n" in response else [response]
    task_list = [{"task_name": task_name} for task_name in new_tasks]
    new_tasks_list: list[str] = []
    for task_item in task_list:
        # print(task_item)
        task_description = task_item.get("task_name")
        if task_description:
            # print(task_description)
            task_parts = task_description.strip().split(".", 1)
            # print(task_parts)
            if len(task_parts) == 2:
                new_task = task_parts[1].strip()
                new_tasks_list.append(new_task)

    return new_tasks_list[0:1]


@traceable(run_type="chain")
def run_agent(main_objective: str, initial_task: str = "") -> list[dict[str, str]]:
    generated_tasks = []
    expounded_initial_task = expound_task(main_objective, initial_task)
    new_tasks = generate_tasks(main_objective, expounded_initial_task)
    print(f"Generated {len(new_tasks)} tasks")
    task_counter = 0
    for task in new_tasks:
        task_counter += 1
        print(f"#### ({task_counter}) Generated Task ####")
        q = expound_task(main_objective, task)
        exp = dump_task(q)
        generated_tasks.append({f"task_{task_counter}": exp})
        if task_counter >= LIMIT:
            print(f"Stopping after {LIMIT} tasks")
            break
    return generated_tasks


if __name__ == "__main__":
    main_objective = "Become a machine learning expert."  # overall objective
    initial_task = "Learn about tensors."  # first task to research

    print("*****OBJECTIVE*****")
    print(f"{main_objective}")

    # Simple version here, just generate tasks based on the initial task and objective,
    # then expound with GPT against the main objective and the newly generated tasks.
    result = run_agent(main_objective, initial_task)
    print("*****RESULT*****")
    print(result)


# # Create a ContextVar to store the current trace context
# _TRACE_CONTEXT = contextvars.ContextVar("_TRACE_CONTEXT", default=None)
#
#
# def get_trace_context() -> Optional[str]:
#     """Get the current trace context."""
#     return _TRACE_CONTEXT.get()
#
#
# def log_decorator(func):
#     @functools.wraps(func)
#     def wrapper(*args, **kwargs):
#         # Get the parent trace_id from the current context
#         parent_trace_id = get_trace_context()
#
#         # Generate a new trace_id for this function call
#         trace_id = gen_trace_id()
#
#         # Set the new trace_id as the current context
#         _TRACE_CONTEXT.set(trace_id)
#         trace_name = func.__name__
#
#         start_timestamp = time.time()
#
#         # Get function arguments using inspect
#         arg_names = inspect.getfullargspec(func)[0]
#         print(f"arg_names: {arg_names}")
#         inputs = {**dict(zip(arg_names, args)), **kwargs}
#         print(f"inputs: {inputs}")
#
#         try:
#             partial_func = functools.partial(func, trace_id=trace_id, trace_name=trace_name, metadata={"parent_trace_id": parent_trace_id})
#             result = partial_func(*args, **kwargs)
#         except Exception as e:
#             # Log the error and re-raise
#             end_timestamp = time.time()
#             log_chain(
#                 trace_id=trace_id,
#                 trace_name=trace_name,
#                 llm_inputs=inputs,
#                 output=None,
#                 error=str(e),
#                 start_timestamp=start_timestamp,
#                 end_timestamp=end_timestamp,
#                 parent_trace_id=parent_trace_id,
#             )
#             raise
#         else:
#             # Log the successful execution
#             end_timestamp = time.time()
#             log_chain(
#                 trace_id=trace_id,
#                 trace_name=trace_name,
#                 llm_inputs=inputs,
#                 output=result,
#                 error=None,
#                 start_timestamp=start_timestamp,
#                 end_timestamp=end_timestamp,
#                 parent_trace_id=parent_trace_id,
#             )
#             return result
#         finally:
#             # Reset the trace context to the parent trace_id
#             _TRACE_CONTEXT.set(parent_trace_id)
#
#     return wrapper
#
#
# def to_date_and_time_string(timestamp: float) -> str:
#     return datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S %Z")
#
#
# def log_chain(
#     trace_id: str,
#     trace_name: str,
#     llm_inputs: Optional[dict],
#     output: Optional[str],
#     error: Optional[str],
#     start_timestamp: float,
#     end_timestamp: float,
#     parent_trace_id: Optional[str],
# ) -> None:
#     httpx.post(
#         "http://localhost:8000/api/parea/v1/log",
#         json={
#             "trace_id": trace_id,
#             "name": trace_name,
#             "error": error,
#             "status": "success" if error is None else "error",
#             "llm_inputs": llm_inputs,
#             "output": output,
#             "start_timestamp": to_date_and_time_string(start_timestamp),
#             "end_timestamp": to_date_and_time_string(end_timestamp),
#             "user_defined_metadata": {"parent_trace_id": parent_trace_id},
#         },
#         headers={"x-api-key": os.getenv("DEV_API_KEY")},
#         timeout=60 * 3.0,
#     )
#
#
# # dump task array to string
# def dump_task(task):
#     d = ""  # init
#     for tasklet in task:
#         d += f"\n{tasklet.get('task_name','')}"
#     d = d.strip()
#     return d
#
#
# def open_ai_inference(
#     prompt: str,
#     model: str = "gpt-3.5-turbo",
#     temperature: float = 0.5,
#     max_tokens: int = 1024,
#     **kwargs,
# ) -> str:
#     return p.completion(
#         data=Completion(
#             llm_configuration=LLMInputs(
#                 model=model,
#                 provider="openai",
#                 model_params=ModelParams(model=model, temp=temperature, max_length=max_tokens),
#                 messages=[Message(role=Role.system, content=prompt)],
#             ),
#             **kwargs,
#         )
#     ).content.strip()
#
#
# @log_decorator
# def expound_task(main_objective: str, current_task: str, **kwargs) -> List[Dict[str, str]]:
#     print(f"****Expounding based on task:**** {current_task}")
#     prompt = f"You are an AI who performs one task based on the following objective: {main_objective}\n" f"Your task: {current_task}\nResponse:"
#     response = open_ai_inference(prompt, name="expound_task", **kwargs)
#     new_tasks = response.split("\n") if "\n" in response else [response]
#     return [{"task_name": task_name} for task_name in new_tasks]
#
#
# @log_decorator
# def generate_tasks(main_objective: str, expounded_initial_task: List[Dict[str, str]], **kwargs) -> List[str]:
#     task_expansion = dump_task(expounded_initial_task)
#     prompt = (
#         f"You are an AI who creates tasks based on the following MAIN OBJECTIVE: {main_objective}\n"
#         f"Create tasks pertaining directly to your previous research here:\n"
#         f"{task_expansion}\nResponse:"
#     )
#     response = open_ai_inference(prompt, name="generate_tasks", **kwargs)
#     new_tasks = response.split("\n") if "\n" in response else [response]
#     task_list = [{"task_name": task_name} for task_name in new_tasks]
#     new_tasks_list: List[str] = []
#     for task_item in task_list:
#         # print(task_item)
#         task_description = task_item.get("task_name")
#         if task_description:
#             # print(task_description)
#             task_parts = task_description.strip().split(".", 1)
#             # print(task_parts)
#             if len(task_parts) == 2:
#                 new_task = task_parts[1].strip()
#                 new_tasks_list.append(new_task)
#
#     return new_tasks_list
#
#
# @log_decorator
# def run_agent(main_objective: str, initial_task: str = "", **kwargs) -> List[Dict[str, str]]:
#     generated_tasks = []
#     expounded_initial_task = expound_task(main_objective, initial_task, **kwargs)
#     new_tasks = generate_tasks(main_objective, expounded_initial_task, **kwargs)
#     print(f"Generated {len(new_tasks)} tasks")
#     task_counter = 0
#     for task in new_tasks:
#         task_counter += 1
#         print(f"#### ({task_counter}) Generated Task ####")
#         q = expound_task(main_objective, task, **kwargs)
#         exp = dump_task(q)
#         generated_tasks.append({f"task_{task_counter}": exp})
#         if task_counter >= LIMIT:
#             print(f"Stopping after {LIMIT} tasks")
#             break
#     return generated_tasks
#
#
# if __name__ == "__main__":
#     main_objective = "Become a machine learning expert."  # overall objective
#     initial_task = "Learn about tensors."  # first task to research
#
#     print("*****OBJECTIVE*****")
#     print(f"{main_objective}")
#
#     # Simple version here, just generate tasks based on the initial task and objective,
#     # then expound with GPT against the main objective and the newly generated tasks.
#     result = run_agent(main_objective, initial_task)
#     print("*****RESULT*****")
#     print(result)
