import asyncio
import os

from dotenv import load_dotenv

from parea.client import Parea
from parea.schemas.models import Completion, CompletionResponse, UseDeployedPrompt, UseDeployedPromptResponse

load_dotenv()

p = Parea(api_key=os.getenv("API_KEY"))

# You will find this deployment_id in the Parea dashboard
deployment_id = os.getenv("DEPLOYMENT_ID")
# Assuming your deployed prompt's message is:
# {"role": "user", "content": "Write a hello world program using {{x}} and the {{y}} framework."}
inputs = {"inputs": {"x": "Golang", "y": "Fiber"}}

# You can easily unpack a dictionary into an attrs class
test_completion = Completion(**{"deployment_id": deployment_id, "llm_inputs": inputs, "metadata": {"purpose": "testing"}})

# By passing in my inputs, in addition to the raw message with unfilled variables {{x}} and {{y}},
# you we will also get the filled-in prompt:
# {"role": "user", "content": "Write a hello world program using Golang and the Fiber framework."}
test_get_prompt = UseDeployedPrompt(deployment_id=deployment_id, inputs=inputs)


def main():
    completion_response: CompletionResponse = p.completion(data=test_completion)
    print(completion_response)
    deployed_prompt: UseDeployedPromptResponse = p.get_prompt(data=test_get_prompt)
    print("\n\n")
    print(deployed_prompt)


async def main_async():
    completion_response: CompletionResponse = await p.acompletion(data=test_completion)
    print(completion_response)
    deployed_prompt: UseDeployedPromptResponse = await p.aget_prompt(data=test_get_prompt)
    print("\n\n")
    print(deployed_prompt)


def hello(name: str) -> str:
    return f"Hello {name}!"


if __name__ == "__main__":
    main()
    asyncio.run(main_async())
