import os
import random
import time

from dotenv import load_dotenv

from parea import Parea
from parea.schemas.models import Completion, CompletionResponse, FeedbackRequest, LLMInputs, Message, ModelParams
from parea.utils.trace_utils import get_current_trace_id, trace

load_dotenv()

p = Parea(api_key=os.getenv("DEV_API_KEY"))


LIMIT = 1


def dump_task(task):
    d = ""
    for tasklet in task:
        d += f"\n{tasklet.get('task_name','')}"
    d = d.strip()
    return d


@trace
def call_llm(
    data: list[dict],
    model: str = "gpt-3.5-turbo",
    provider: str = "openai",
    temperature: float = 0.0,
) -> CompletionResponse:
    return p.completion(
        data=Completion(
            llm_configuration=LLMInputs(
                model=model,
                provider=provider,
                model_params=ModelParams(temp=temperature),
                messages=[Message(**d) for d in data],
            )
        )
    )


@trace
def expound_task(main_objective: str, current_task: str) -> list[dict[str, str]]:
    prompt = [
        {
            "role": "system",
            "content": f"You are an AI who performs one task based on the following objective: {main_objective}\n" f"Your task: {current_task}\nResponse:",
        },
    ]
    response = call_llm(prompt).content
    new_tasks = response.split("\n") if "\n" in response else [response]
    return [{"task_name": task_name} for task_name in new_tasks]


@trace
def generate_tasks(main_objective: str, expounded_initial_task: list[dict[str, str]]) -> list[str]:
    llm_options = [("gpt-3.5-turbo", "openai"), ("gpt-4", "openai"), ("claude-instant-1", "anthropic"), ("claude-2", "anthropic")]
    select_llm_option = random.choice(llm_options)

    task_expansion = dump_task(expounded_initial_task)
    prompt = [
        {
            "role": "system",
            "content": (
                f"You are an AI who creates tasks based on the following MAIN OBJECTIVE: {main_objective}\n"
                f"Create tasks pertaining directly to your previous research here:\n"
                f"{task_expansion}\nResponse:"
            ),
        },
    ]
    response = call_llm(data=prompt, model=select_llm_option[0], provider=select_llm_option[1]).content
    new_tasks = response.split("\n") if "\n" in response else [response]
    task_list = [{"task_name": task_name} for task_name in new_tasks]
    new_tasks_list: list[str] = []
    for task_item in task_list:
        task_description = task_item.get("task_name")
        if task_description:
            task_parts = task_description.strip().split(".", 1)
            if len(task_parts) == 2:
                new_task = task_parts[1].strip()
                new_tasks_list.append(new_task)

    return new_tasks_list


@trace
def run_agent(main_objective: str, initial_task: str = "") -> tuple[list[dict[str, str]], str]:
    trace_id = get_current_trace_id()
    generated_tasks = []
    expounded_initial_task = expound_task(main_objective, initial_task)
    new_tasks = generate_tasks(main_objective, expounded_initial_task)
    task_counter = 0
    for task in new_tasks or []:
        task_counter += 1
        q = expound_task(main_objective, task)
        exp = dump_task(q)
        generated_tasks.append({f"task_{task_counter}": exp})
        if task_counter >= LIMIT:
            break
    return generated_tasks, trace_id


if __name__ == "__main__":
    result, trace_id = run_agent("Become a machine learning expert.", "Learn about tensors.")
    time.sleep(3)
    p.record_feedback(FeedbackRequest(trace_id=trace_id, score=0.642))
    print(result)
