from typing import Dict, Optional

import os
import random
from datetime import datetime

import openai
from dotenv import load_dotenv

from parea import Parea
from parea.schemas.models import FeedbackRequest
from parea.utils.trace_utils import get_current_trace_id, trace

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

p = Parea(api_key=os.getenv("PAREA_API_KEY"))


def call_llm(data: list[dict], model: str = "gpt-3.5-turbo", temperature: float = 0.0) -> str:
    return openai.ChatCompletion.create(model=model, temperature=temperature, messages=data).choices[0].message["content"]


def random_eval(inputs: Dict[str, str], output, target: Optional[str] = None) -> float:
    # return random number between 0 and 1
    return random.random()


@trace(eval_funcs=[random_eval])
def argumentor(query: str, additional_description: str = "") -> str:
    return call_llm(
        [
            {
                "role": "system",
                "content": f"""You are a debater making an argument on a topic. {additional_description}.
                The current time is {datetime.now().strftime("%Y-%m-%d")}""",
            },
            {"role": "user", "content": f"The discussion topic is {query}"},
        ]
    )


@trace
def critic(argument: str) -> str:
    return call_llm(
        [
            {
                "role": "system",
                "content": f"""You are a critic.
                What unresolved questions or criticism do you have after reading the following argument?
                Provide a concise summary of your feedback.""",
            },
            {"role": "user", "content": argument},
        ]
    )


@trace(eval_funcs=[random_eval])
def refiner(query: str, additional_description: str, argument: str, criticism: str) -> str:
    return call_llm(
        [
            {
                "role": "system",
                "content": f"""You are a debater making an argument on a topic. {additional_description}.
                The current time is {datetime.now().strftime("%Y-%m-%d")}""",
            },
            {"role": "user", "content": f"""The discussion topic is {query}"""},
            {"role": "assistant", "content": argument},
            {"role": "user", "content": criticism},
            {
                "role": "system",
                "content": f"Please generate a new argument that incorporates the feedback from the user.",
            },
        ],
    )


@trace(eval_funcs=[random_eval], access_output_of_func=lambda x: x[0])
def argument_chain(query: str, additional_description: str = "") -> tuple[str, str]:
    trace_id = get_current_trace_id()
    argument = argumentor(query, additional_description)
    criticism = critic(argument)
    refined_argument = refiner(query, additional_description, argument, criticism)
    return refined_argument, trace_id


if __name__ == "__main__":
    result, trace_id = argument_chain(
        "Whether sparkling wine is good for you.",
        additional_description="Provide a concise, few sentence argument on why sparkling wine is good for you.",
    )
    print(result)
    p.record_feedback(
        FeedbackRequest(
            trace_id=trace_id,
            score=0.7,  # 0.0 (bad) to 1.0 (good)
        )
    )
