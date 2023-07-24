import asyncio
import os

from dotenv import load_dotenv

from parea.client import Parea
from parea.schemas.models import Completion, UseDeployedPrompt

load_dotenv()

p = Parea(api_key=os.getenv("API_KEY"))

# You will find this deployment_id in the Parea dashboard
deployment_id = "p-qsefFeFEICnxqJ_yLjji"
# Assuming my deployed prompt's message is:
# {"role": "user", "content": "Write a hello world program using {{x}} and the {{y}} framework."}
inputs = {"inputs": {"x": "Golang", "y": "Fiber"}}
test_completion = Completion(**{"deployment_id": deployment_id, "llm_inputs": inputs, "metadata": {"purpose": "testing"}})
# By passing in my inputs, instead of unfilled variables {{x}} and {{y}}, we will also have the filled in prompt:
# {"role": "user", "content": "Write a hello world program using Golang and the Fiber framework."}
test_get_prompt = UseDeployedPrompt(deployment_id, inputs)


def main():
    r = p.completion(data=test_completion)
    print(r)
    r2 = p.get_prompt(data=test_get_prompt)
    print("\n\n")
    print(r2)


async def main_async():
    r = await p.acompletion(data=test_completion)
    print(r)
    r2 = await p.aget_prompt(data=test_get_prompt)
    print("\n\n")
    print(r2)


def hello(name: str) -> str:
    return f"Hello {name}!"


if __name__ == "__main__":
    main()
    asyncio.run(main_async())
