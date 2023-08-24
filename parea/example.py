import asyncio
import os
from datetime import datetime

from dotenv import load_dotenv

from parea.client import Parea
from parea.schemas.models import Completion, CompletionResponse, UseDeployedPrompt, UseDeployedPromptResponse
from parea.trace_utils.trace import trace_insert, traceable

load_dotenv()

p = Parea(api_key=os.getenv("DEV_API_KEY"))


@traceable
def main(llm_inputs, metadata):
    # You will find this deployment_id in the Parea dashboard
    deployment_id = os.getenv("DEV_DEPLOYMENT_ID")

    # You can easily unpack a dictionary into an attrs class
    test_completion = Completion(**{"deployment_id": deployment_id, "llm_inputs": llm_inputs, "metadata": metadata})
    completion_response: CompletionResponse = p.completion(data=test_completion)
    print(completion_response)

    # By passing in my inputs, in addition to the raw message with unfilled variables {{x}} and {{y}},
    # you we will also get the filled-in prompt:
    # {"role": "user", "content": "Write a hello world program using Golang and the Fiber framework."}
    test_get_prompt = UseDeployedPrompt(deployment_id=deployment_id, llm_inputs=llm_inputs)
    deployed_prompt: UseDeployedPromptResponse = p.get_prompt(data=test_get_prompt)
    print("\n\n")
    return completion_response, deployed_prompt


# @traceable
# async def main_async():
#     completion_response: CompletionResponse = await p.acompletion(data=test_completion)
#     print(completion_response)
#     deployed_prompt: UseDeployedPromptResponse = await p.aget_prompt(data=test_get_prompt)
#     print("\n\n")
#     print(deployed_prompt)

#
# def hello(name: str) -> str:
#     return f"Hello {name}!"


@traceable
def run_child1(x):
    return 1 + x


@traceable
def run_child2(y):
    return run_grand_child1(y) + y


@traceable
def run_grand_child1(z):
    # Add metadata to the trace data for this function
    trace_insert({"metadata": {"internal": True, "tokens": 3}})
    return 3 * z


@traceable
def parent(x, y):
    answer1 = run_child1(x)
    answer2 = run_child2(y)
    return (answer1 + answer2) / 2


@traceable
def parent2(x, y):
    return (x + y) / 2


@traceable
async def arun_child1(x):
    await asyncio.sleep(1)  # simulate IO-bound operation
    return 1 + x


@traceable
async def arun_child2(y):
    res = await arun_grand_child1(y)
    return res + y


@traceable
async def arun_grand_child1(z):
    await asyncio.sleep(1)  # simulate IO-bound operation
    trace_insert({"metadata": {"internal": True, "tokens": 3}})
    return 3 * z


@traceable
async def aparent(x, y):
    answer1 = await arun_child1(x)
    answer2 = await arun_child2(y)
    return (answer1 + answer2) / 2


@traceable
async def aparent2(x, y):
    return (x + y) / 2


def retester_main():
    parent(6, 2)
    parent(3, 4)
    parent2(3, 12)


async def atester_main():
    await aparent(6, 2)
    await aparent(3, 4)
    await aparent2(3, 12)


@traceable
def argument_generator(query: str, additional_description: str = "") -> str:
    return p.completion(
        Completion(
            deployment_id="p-ee6hsbh1_4WtIz6UO6rEA",
            llm_inputs={
                "additional_description": additional_description,
                "date": f"{datetime.now()}",
                "query": query,
            },
        )
    ).content


@traceable
def critic(argument: str) -> str:
    return p.completion(
        Completion(
            deployment_id="p-_4w8th0eJlWAx9g8g20Ov",
            llm_inputs={"argument": argument},
            end_user_identifier="user_1",
            metadata={"githash": "e38f04c83"},
            tags=["tutorial"],
            target="you're right, i give up",
        )
    ).content


@traceable
def refiner(query: str, additional_description: str, current_arg: str, criticism: str) -> str:
    return p.completion(
        Completion(
            deployment_id="p-k2Iuvik12X_Uk4y-fOFPp",
            llm_inputs={
                "additional_description": additional_description,
                "date": f"{datetime.now()}",
                "query": query,
                "current_arg": current_arg,
                "criticism": criticism,
            },
        )
    ).content


@traceable
def argument_chain(query: str, additional_description: str = "") -> str:
    argument = argument_generator(query, additional_description)
    criticism = critic(argument)
    return refiner(query, additional_description, argument, criticism)


if __name__ == "__main__":
    result = argument_chain(
        "Whether sunshine is good for you.",
        additional_description="Provide a concise, few sentence argument on why sunshine is good for you.",
    )
    # retester_main()
