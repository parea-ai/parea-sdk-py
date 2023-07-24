import asyncio
import os

from dotenv import load_dotenv

from parea.client import Parea
from parea.schemas.models import Completion, UseDeployedPrompt

load_dotenv()

p = Parea(api_key=os.getenv("DEV_API_KEY"))

deployment_id = os.getenv("DEV_DEPLOYMENT_ID")
inputs = {"inputs": {"x": "Golang", "y": "Fiber"}}
test_completion = Completion(**{"prompt_deployment_id": deployment_id, "llm_inputs": inputs, "metadata": {"purpose": "testing"}})
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
