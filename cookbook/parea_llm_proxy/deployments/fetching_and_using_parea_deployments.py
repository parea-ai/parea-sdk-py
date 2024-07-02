import os

from dotenv import load_dotenv

from parea import Parea
from parea.schemas.models import Completion, CompletionResponse, UseDeployedPrompt, UseDeployedPromptResponse

load_dotenv()

p = Parea(api_key=os.getenv("PAREA_API_KEY"))


def main() -> CompletionResponse:
    return p.completion(Completion(deployment_id="p-4cbYJ0LIy0gaWb6Z819k7", llm_inputs={"x": "python", "y": "fastapi"}))


def get_critic_prompt(val: str) -> UseDeployedPromptResponse:
    return p.get_prompt(UseDeployedPrompt(deployment_id="p-87NFVeQg30Hk2Hatw1h72", llm_inputs={"x": val}))


if __name__ == "__main__":
    print(get_critic_prompt("Python"))
    # a = UseDeployedPromptResponse(
    #     deployment_id="p-87NFVeQg30Hk2Hatw1h72",
    #     name="deploy-test",
    #     functions=[],
    #     function_call=None,
    #     prompt=Prompt(
    #         raw_messages=[{"role": "user", "content": "Write a hello world program in {{x}}"}],
    #         messages=[{"content": "Write a hello world program in Python", "role": "user"}],
    #         inputs={"x": "Python"},
    #     ),
    #     model="gpt-3.5-turbo-0125",
    #     provider="openai",
    #     model_params={"temp": 0.0, "top_p": 1.0, "max_length": None, "presence_penalty": 0.0, "frequency_penalty": 0.0},
    # )
