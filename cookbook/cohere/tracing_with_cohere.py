from typing import List, Optional

import os
from datetime import datetime

import cohere
from dotenv import load_dotenv

from parea import Parea, trace, trace_insert

load_dotenv()

p = Parea(api_key=os.getenv("PAREA_API_KEY"))
co = cohere.Client(api_key=os.getenv("COHERE_API_KEY"))
p.wrap_cohere_client(co)


def call_llm(message: str, chat_history: Optional[List[dict]] = None, system_message: str = "", model: str = "command-r-plus") -> str:
    return co.chat(
        model=model,
        preamble=system_message,
        chat_history=chat_history or [],
        message=message,
    ).text


@trace
def argumentor(query: str, additional_description: str = "") -> str:
    return call_llm(
        system_message=f"""You are a debater making an argument on a topic. {additional_description}.
        The current time is {datetime.now().strftime("%Y-%m-%d")}""",
        message=f"The discussion topic is {query}",
    )


@trace
def critic(argument: str) -> str:
    return call_llm(
        system_message="""You are a critic.
                What unresolved questions or criticism do you have after reading the following argument?
                Provide a concise summary of your feedback.""",
        message=argument,
    )


@trace
def refiner(query: str, additional_description: str, argument: str, criticism: str) -> str:
    return call_llm(
        system_message=f"""You are a debater making an argument on a topic. {additional_description}.
                The current time is {datetime.now().strftime("%Y-%m-%d")}""",
        chat_history=[{"role": "USER", "message": f"""The discussion topic is {query}"""}, {"role": "CHATBOT", "message": argument}, {"role": "USER", "message": criticism}],
        message="Please generate a new argument that incorporates the feedback from the user.",
    )


@trace
def argument_chain(query: str, additional_description: str = "") -> str:
    trace_insert({"session_id": "cus_1234", "end_user_identifier": "user_1234"})
    argument = argumentor(query, additional_description)
    criticism = critic(argument)
    refined_argument = refiner(query, additional_description, argument, criticism)
    return refined_argument


@trace(session_id="cus_1234", end_user_identifier="user_1234")
def json_call() -> str:
    completion = co.chat(
        model="command-r-plus",
        preamble="You are a helpful assistant talking in JSON.",
        message="What are you?",
        response_format={"type": "json_object"},
    )
    return completion.text


if __name__ == "__main__":
    result = argument_chain(
        "Whether sparkling wine is good for you.",
        additional_description="Provide a concise, few sentence argument on why sparkling wine is good for you.",
    )
    # print(result)
    # print(json_call())
