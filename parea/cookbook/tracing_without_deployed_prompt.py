import os
from datetime import datetime

from dotenv import load_dotenv

from parea import Parea
from parea.schemas.log import LLMInputs, Message, ModelParams
from parea.schemas.models import Completion, CompletionResponse, FeedbackRequest
from parea.utils.trace_utils import get_current_trace_id, trace

load_dotenv()

p = Parea(api_key=os.getenv("PAREA_API_KEY"))


@trace  # <--- If you want to log the inputs to the LLM call you can optionally add a trace decorator here
def call_llm(
    data: list[dict],
    model: str = "gpt-3.5-turbo",
    provider: str = "openai",
    temperature: float = 0.0,
) -> CompletionResponse:
    return p.completion(
        data=Completion(
            llm_configuration=LLMInputs(
                model=model,
                provider=provider,
                model_params=ModelParams(temp=temperature),
                messages=[Message(**d) for d in data],
            )
        )
    )


def argument_generator(query: str, additional_description: str = "") -> str:
    return call_llm(
        [
            {
                "role": "system",
                "content": f"You are a debater making an argument on a topic." f"{additional_description}" f" The current time is {datetime.now()}",
            },
            {"role": "user", "content": f"The discussion topic is {query}"},
        ]
    ).content


def critic(argument: str) -> str:
    return call_llm(
        [
            {
                "role": "system",
                "content": f"You are a critic."
                "\nWhat unresolved questions or criticism do you have after reading the following argument?"
                "Provide a concise summary of your feedback.",
            },
            {"role": "system", "content": argument},
        ]
    ).content


def refiner(query: str, additional_description: str, current_arg: str, criticism: str) -> str:
    return call_llm(
        [
            {
                "role": "system",
                "content": f"You are a debater making an argument on a topic. {additional_description}. " f"The current time is {datetime.now()}",
            },
            {"role": "user", "content": f"The discussion topic is {query}"},
            {"role": "assistant", "content": current_arg},
            {"role": "user", "content": criticism},
            {
                "role": "system",
                "content": "Please generate a new argument that incorporates the feedback from the user.",
            },
        ]
    ).content


@trace
def argument_chain(query: str, additional_description: str = "") -> str:
    argument = argument_generator(query, additional_description)
    criticism = critic(argument)
    return refiner(query, additional_description, argument, criticism)


@trace
def argument_chain2(query: str, additional_description: str = "") -> tuple[str, str]:
    trace_id = get_current_trace_id()
    argument = argument_generator(query, additional_description)
    criticism = critic(argument)
    return refiner(query, additional_description, argument, criticism), trace_id


def refiner2(query: str, additional_description: str, current_arg: str, criticism: str) -> CompletionResponse:
    return call_llm(
        [
            {
                "role": "system",
                "content": f"You are a debater making an argument on a topic. {additional_description}. The current time is {datetime.now()}",
            },
            {"role": "user", "content": f"The discussion topic is {query}"},
            {"role": "assistant", "content": current_arg},
            {"role": "user", "content": criticism},
            {
                "role": "system",
                "content": "Please generate a new argument that incorporates the feedback from the user.",
            },
        ],
        model="claude-2",
        provider="anthropic",
    )


@trace(
    tags=["cookbook-example", "feedback_tracked"],
    metadata={"source": "python-sdk"},
)
def argument_chain3(query: str, additional_description: str = "") -> CompletionResponse:
    argument = argument_generator(query, additional_description)
    criticism = critic(argument)
    return refiner2(query, additional_description, argument, criticism)


if __name__ == "__main__":
    result1 = argument_chain(
        "Whether coffee is good for you.",
        additional_description="Provide a concise, few sentence argument on why coffee is good for you.",
    )
    print(result1)

    result2, trace_id2 = argument_chain2(
        "Whether wine is good for you.",
        additional_description="Provide a concise, few sentence argument on why wine is good for you.",
    )
    print(result2)
    p.record_feedback(
        FeedbackRequest(
            trace_id=trace_id2,
            score=0.0,  # 0.0 (bad) to 1.0 (good)
            target="Moonshine is wonderful.",
        )
    )

    result3 = argument_chain3(
        "Whether moonshine is good for you.",
        additional_description="Provide a concise, few sentence argument on why moonshine is good for you.",
    )
    print(result3.content)
    p.record_feedback(
        FeedbackRequest(
            trace_id=result3.inference_id,
            score=0.7,  # 0.0 (bad) to 1.0 (good)
            target="Moonshine is wonderful. End of story.",
        )
    )
