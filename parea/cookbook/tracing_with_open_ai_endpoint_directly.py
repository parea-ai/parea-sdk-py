import os
from datetime import datetime

from dotenv import load_dotenv
from openai import OpenAI

from parea import Parea, get_current_trace_id, trace
from parea.schemas import FeedbackRequest

load_dotenv()

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

p = Parea(api_key=os.getenv("PAREA_API_KEY"))
p.wrap_openai_client(client)


def call_llm(data: list[dict], model: str = "gpt-3.5-turbo", temperature: float = 0.0) -> str:
    return client.chat.completions.create(model=model, temperature=temperature, messages=data).choices[0].message.content


@trace
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


@trace
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


@trace
def argument_chain(query: str, additional_description: str = "") -> tuple[str, str]:
    trace_id = get_current_trace_id()
    argument = argumentor(query, additional_description)
    criticism = critic(argument)
    refined_argument = refiner(query, additional_description, argument, criticism)
    return refined_argument, trace_id


@trace
def json_call() -> str:
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo-0125",
        messages=[{"role": "system", "content": "You are a helpful assistant talking in JSON."}, {"role": "user", "content": "What are you?"}],
        response_format={"type": "json_object"},
    )
    return completion.choices[0].message.content


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

    print(json_call())
