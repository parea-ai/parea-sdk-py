import os
import time
from datetime import datetime

from dotenv import load_dotenv

from parea import Parea
from parea.schemas.models import Completion, CompletionResponse, FeedbackRequest, LLMInputs, Message, ModelParams
from parea.utils.trace_utils import get_current_trace_id, trace

load_dotenv()

p = Parea(api_key=os.getenv("DEV_API_KEY"))


LIMIT = 10


def dump_task(task):
    d = ""
    for tasklet in task:
        d += f"\n{tasklet.get('task_name','')}"
    d = d.strip()
    return d


@trace
def call_openai(
    data: list[dict],
    model: str = "gpt-3.5-turbo",
    temperature: float = 0.0,
) -> CompletionResponse:
    return p.completion(
        data=Completion(
            llm_configuration=LLMInputs(
                model=model,
                provider="openai",
                model_params=ModelParams(temp=temperature),
                messages=[Message(**d) for d in data],
            )
        )
    )


@trace
def argument_generator(query: str, additional_description: str = "") -> str:
    return call_openai(
        [
            {
                "role": "system",
                "content": f"You are a debater making an argument on a topic." f"{additional_description}" f" The current time is {datetime.now()}",
            },
            {"role": "user", "content": f"The discussion topic is {query}"},
        ]
    ).content


@trace
def critic(argument: str) -> str:
    return call_openai(
        [
            {
                "role": "system",
                "content": f"You are a critic."
                "\nWhat unresolved questions or criticism do you have after reading the following argument?"
                "Provide a concise summary of your feedback.",
            },
            {"role": "system", "content": argument},
        ]
    ).content


@trace
def refiner(query: str, additional_description: str, current_arg: str, criticism: str) -> str:
    return call_openai(
        [
            {
                "role": "system",
                "content": f"You are a debater making an argument on a topic. {additional_description}. " f"The current time is {datetime.now()}",
            },
            {"role": "user", "content": f"The discussion topic is {query}"},
            {"role": "assistant", "content": current_arg},
            {"role": "user", "content": criticism},
            {
                "role": "system",
                "content": "Please generate a new argument that incorporates the feedback from the user.",
            },
        ]
    ).content


@trace
def deployed_argument_generator(query: str, additional_description: str = "") -> str:
    return p.completion(
        Completion(
            deployment_id="p-Ar-Oi14-nBxHUiradyql9",
            llm_inputs={
                "additional_description": additional_description,
                "date": f"{datetime.now()}",
                "query": query,
            },
        )
    ).content


@trace
def deployed_critic(argument: str) -> str:
    return p.completion(
        Completion(
            deployment_id="p-W2yPy93tAczYrxkipjli6",
            llm_inputs={"argument": argument},
        )
    ).content


@trace
def deployed_refiner(query: str, additional_description: str, current_arg: str, criticism: str) -> str:
    return p.completion(
        Completion(
            deployment_id="p-8Er1Xo0GDGF2xtpmMOpbn",
            llm_inputs={
                "additional_description": additional_description,
                "date": f"{datetime.now()}",
                "query": query,
                "current_arg": current_arg,
                "criticism": criticism,
            },
        )
    ).content


@trace
def argument_chain(query: str, additional_description: str = "") -> str:
    argument = argument_generator(query, additional_description)
    criticism = critic(argument)
    return refiner(query, additional_description, argument, criticism)


@trace
def argument_chain2(query: str, additional_description: str = "") -> tuple[str, str]:
    trace_id = get_current_trace_id()
    argument = argument_generator(query, additional_description)
    criticism = critic(argument)
    return refiner(query, additional_description, argument, criticism), trace_id


@trace
def refiner2(query: str, additional_description: str, current_arg: str, criticism: str) -> CompletionResponse:
    return call_openai(
        [
            {
                "role": "system",
                "content": f"You are a debater making an argument on a topic. {additional_description}. The current time is {datetime.now()}",
            },
            {"role": "user", "content": f"The discussion topic is {query}"},
            {"role": "assistant", "content": current_arg},
            {"role": "user", "content": criticism},
            {
                "role": "system",
                "content": "Please generate a new argument that incorporates the feedback from the user.",
            },
        ]
    )


@trace
def deployed_refiner2(query: str, additional_description: str, current_arg: str, criticism: str) -> CompletionResponse:
    return p.completion(
        Completion(
            deployment_id="p-8Er1Xo0GDGF2xtpmMOpbn",
            llm_inputs={
                "additional_description": additional_description,
                "date": f"{datetime.now()}",
                "query": query,
                "current_arg": current_arg,
                "criticism": criticism,
            },
        )
    )


@trace(
    tags=["cookbook-example", "feedback_tracked"],
    metadata={"source": "python-sdk"},
)
def argument_chain3(query: str, additional_description: str = "") -> CompletionResponse:
    argument = argument_generator(query, additional_description)
    criticism = critic(argument)
    return refiner2(query, additional_description, argument, criticism)


@trace
def deployed_argument_chain(query: str, additional_description: str = "") -> str:
    argument = deployed_argument_generator(query, additional_description)
    criticism = deployed_critic(argument)
    return deployed_refiner(query, additional_description, argument, criticism)


@trace
def deployed_argument_chain2(query: str, additional_description: str = "") -> tuple[str, str]:
    trace_id = get_current_trace_id()
    argument = deployed_argument_generator(query, additional_description)
    criticism = deployed_critic(argument)
    return deployed_refiner(query, additional_description, argument, criticism), trace_id


@trace(
    tags=["cookbook-example-deployed", "feedback_tracked-deployed"],
    metadata={"source": "python-sdk", "deployed": True},
)
def deployed_argument_chain3(query: str, additional_description: str = "") -> CompletionResponse:
    argument = deployed_argument_generator(query, additional_description)
    criticism = deployed_critic(argument)
    return deployed_refiner2(query, additional_description, argument, criticism)


@trace
def expound_task(main_objective: str, current_task: str) -> list[dict[str, str]]:
    prompt = [
        {
            "role": "system",
            "content": f"You are an AI who performs one task based on the following objective: {main_objective}\n" f"Your task: {current_task}\nResponse:",
        },
    ]
    response = call_openai(prompt).content
    new_tasks = response.split("\n") if "\n" in response else [response]
    return [{"task_name": task_name} for task_name in new_tasks]


@trace
def generate_tasks(main_objective: str, expounded_initial_task: list[dict[str, str]]) -> list[str]:
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
    response = call_openai(prompt).content
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
    result = argument_chain(
        "Whether coffee is good for you.",
        additional_description="Provide a concise, few sentence argument on why coffee is good for you.",
    )
    print(result)

    result2, trace_id = argument_chain2(
        "Whether wine is good for you.",
        additional_description="Provide a concise, few sentence argument on why wine is good for you.",
    )
    time.sleep(3)
    p.record_feedback(
        FeedbackRequest(
            trace_id=trace_id,
            score=0.0,  # 0.0 (bad) to 1.0 (good)
            target="Moonshine is wonderful.",
        )
    )
    print(result2)

    result3 = argument_chain3(
        "Whether moonshine is good for you.",
        additional_description="Provide a concise, few sentence argument on why moonshine is good for you.",
    )
    time.sleep(3)
    p.record_feedback(
        FeedbackRequest(
            trace_id=result3.inference_id,
            score=0.7,  # 0.0 (bad) to 1.0 (good)
            target="Moonshine is wonderful. End of story.",
        )
    )
    print(result3.content)

    result4, trace_id2 = run_agent("Become a machine learning expert.", "Learn about tensors.")
    time.sleep(3)
    p.record_feedback(
        FeedbackRequest(
            trace_id=trace_id2,
            score=0.642,  # 0.0 (bad) to 1.0 (good)
            target="Do both!.",
        )
    )
    print(result4)

    result5 = deployed_argument_chain(
        "Whether coffee is good for you.",
        additional_description="Provide a concise, few sentence argument on why coffee is good for you.",
    )
    print(result5)

    result6, trace_id3 = deployed_argument_chain2(
        "Whether wine is good for you.",
        additional_description="Provide a concise, few sentence argument on why wine is good for you.",
    )
    time.sleep(3)
    p.record_feedback(
        FeedbackRequest(
            trace_id=trace_id3,
            score=0.0,  # 0.0 (bad) to 1.0 (good)
            target="Moonshine is wonderful.",
        )
    )
    print(result6)

    result7 = deployed_argument_chain3(
        "Whether moonshine is good for you.",
        additional_description="Provide a concise, few sentence argument on why moonshine is good for you.",
    )
    time.sleep(3)
    p.record_feedback(
        FeedbackRequest(
            trace_id=result7.inference_id,
            score=0.7,  # 0.0 (bad) to 1.0 (good)
            target="Moonshine is wonderful. End of story.",
        )
    )
    print(result7.error or result7.content)
