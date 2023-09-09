import os
from datetime import datetime

import openai
from dotenv import load_dotenv

from parea import Parea
from parea.schemas.models import FeedbackRequest
from parea.utils.trace_utils import get_current_trace_id, trace

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

p = Parea(api_key=os.getenv("PAREA_API_KEY"))


@trace
def argument_chain(query: str, additional_description: str = "") -> str:
    trace_id = get_current_trace_id()
    argument = (
        openai.ChatCompletion.create(
            model="gpt-3.5-turbo-0613",
            temperature=0.0,
            messages=[
                {
                    "role": "system",
                    "content": f"""You are a debater making an argument on a topic. {additional_description}.
The current time is {datetime.now()}""",
                },
                {"role": "user", "content": f"The discussion topic is {query}"},
            ],
        )
        .choices[0]
        .message["content"]
    )

    criticism = (
        openai.ChatCompletion.create(
            model="gpt-3.5-turbo-0613",
            temperature=0.0,
            messages=[
                {
                    "role": "system",
                    "content": f"""You are a critic.
What unresolved questions or criticism do you have after reading the following argument?
Provide a concise summary of your feedback.""",
                },
                {"role": "user", "content": argument},
            ],
        )
        .choices[0]
        .message["content"]
    )

    refined_argument = (
        openai.ChatCompletion.create(
            model="gpt-3.5-turbo-0613",
            temperature=0.0,
            messages=[
                {
                    "role": "system",
                    "content": f"""You are a debater making an argument on a topic. {additional_description}.
The current time is {datetime.now()}""",
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
        .choices[0]
        .message["content"]
    )

    return refined_argument, trace_id


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
