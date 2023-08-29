import os
from datetime import datetime

import openai
from dotenv import load_dotenv

from parea import Parea
from parea.schemas.models import FeedbackRequest
from parea.utils.trace_utils import get_current_trace_id, trace

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

p = Parea(api_key=os.getenv("DEV_API_KEY"))


def call_openai(data: list[dict], model: str = "gpt-3.5-turbo-0613", temperature: float = 0.0) -> str:
    return openai.ChatCompletion.create(model=model, messages=data, temperature=temperature).choices[0].message["content"]


@trace
def argument_generator(query: str, additional_description: str = "") -> str:
    return call_openai(
        data=[
            {
                "role": "system",
                "content": f"""You are a debater making an argument on a topic. {additional_description}.
            The current time is {datetime.now()}""",
            },
            {"role": "user", "content": f"""The discussion topic is {query}"""},
        ]
    )


@trace
def critic(argument: str) -> str:
    return call_openai(
        data=[
            {
                "role": "system",
                "content": f"""You are a critic.
                What unresolved questions or criticism do you have after reading the following argument?
                Provide a concise summary of your feedback.""",
            },
            {"role": "user", "content": f"""{argument}"""},
        ]
    )


@trace
def refiner(query: str, additional_description: str, current_arg: str, criticism: str) -> str:
    return call_openai(
        data=[
            {
                "role": "system",
                "content": f"""You are a debater making an argument on a topic. {additional_description}.
                The current time is {datetime.now()}""",
            },
            {"role": "user", "content": f"""The discussion topic is {query}"""},
            {"role": "assistant", "content": f"""{current_arg}"""},
            {"role": "user", "content": f"""{criticism}"""},
            {
                "role": "system",
                "content": f"""Please generate a new argument that incorporates the feedback from the user.""",
            },
        ]
    )


@trace
def argument_chain(query: str, additional_description: str = "") -> tuple[str, str]:
    trace_id = get_current_trace_id()
    argument = argument_generator(query, additional_description)
    criticism = critic(argument)
    return refiner(query, additional_description, argument, criticism), trace_id


if __name__ == "__main__":
    result, trace_id = argument_chain(
        "Whether sparkling water is good for you.",
        additional_description="Provide a concise, few sentence argument on why sparkling water is good for you.",
    )
    print(result)
    p.record_feedback(
        FeedbackRequest(
            trace_id=trace_id,
            score=0.7,  # 0.0 (bad) to 1.0 (good)
        )
    )
