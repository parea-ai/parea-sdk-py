import os
from datetime import datetime

import openai
from dotenv import load_dotenv

from parea import Parea

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

p = Parea(api_key=os.getenv("PAREA_API_KEY"))


def argument_generator(query: str, additional_description: str = "", date=datetime.now()) -> str:
    return (
        openai.ChatCompletion.create(
            model="gpt-3.5-turbo-0613",
            messages=[
                {
                    "role": "system",
                    "content": f"""You are a debater making an argument on a topic.
{additional_description}.
The current time is {date}""",
                },
                {"role": "user", "content": f"""The discussion topic is {query}"""},
            ],
            temperature=0.0,
        )
        .choices[0]
        .message["content"]
    )


def critic(argument: str) -> str:
    return (
        openai.ChatCompletion.create(
            model="gpt-3.5-turbo-0613",
            messages=[
                {
                    "role": "system",
                    "content": f"""You are a critic.
What unresolved questions or criticism do you have after reading the following argument?
Provide a concise summary of your feedback.""",
                },
                {"role": "user", "content": f"""{argument}"""},
            ],
            temperature=0.0,
        )
        .choices[0]
        .message["content"]
    )


def refiner(query: str, additional_description: str, current_arg: str, criticism: str, date=datetime.now()) -> str:
    return (
        openai.ChatCompletion.create(
            model="gpt-3.5-turbo-0613",
            messages=[
                {
                    "role": "system",
                    "content": f"""You are a debater making an argument on a topic.
{additional_description}.
The current time is {date}""",
                },
                {"role": "user", "content": f"""The discussion topic is {query}"""},
                {"role": "assistant", "content": f"""{current_arg}"""},
                {"role": "user", "content": f"""{criticism}"""},
                {"role": "system", "content": f"""Please generate a new argument that incorporates the feedback from the user."""},
            ],
            temperature=0.0,
        )
        .choices[0]
        .message["content"]
    )


def argument_chain(query: str, additional_description: str = "") -> str:
    argument = argument_generator(query, additional_description)
    criticism = critic(argument)
    return refiner(query, additional_description, argument, criticism)


if __name__ == "__main__":
    result = argument_chain(
        "Whether caffeine is good for you.",
        additional_description="Provide a concise, few sentence argument on why caffeine is good for you.",
    )
    print(result)

    from parea.schemas.models import FeedbackRequest
    from parea.utils.trace_utils import get_current_trace_id

    p = Parea(api_key=os.getenv("PAREA_API_KEY"))

    trace_id = get_current_trace_id()
    print(f"trace_id: {trace_id}")
    p.record_feedback(
        FeedbackRequest(
            trace_id=trace_id,
            score=0.7,  # 0.0 (bad) to 1.0 (good)
        )
    )
