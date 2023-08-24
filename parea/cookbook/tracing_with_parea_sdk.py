import os
from datetime import datetime

from dotenv import load_dotenv

from parea import Parea
from parea.schemas.models import Completion, CompletionResponse, FeedbackRequest
from parea.utils.trace_utils import get_current_trace_id, trace

load_dotenv()

p = Parea(api_key=os.getenv("PAREA_API_KEY"))


# We pass the deployment_id and the required inputs to the completion function along with the trace_id
@trace
def argument_generator(query: str, additional_description: str = "", **kwargs) -> str:
    return p.completion(
        Completion(
            deployment_id="p-RG8d9sssc_0cctwfpb_n6",
            llm_inputs={
                "additional_description": additional_description,
                "date": f"{datetime.now()}",
                "query": query,
            },
            **kwargs,
        )
    ).content


@trace
def critic(argument: str, **kwargs) -> str:
    return p.completion(
        Completion(
            deployment_id="p-fXgZytVVVJjXD_71TDR4s",
            llm_inputs={"argument": argument},
            **kwargs,
        )
    ).content


@trace
def refiner(query: str, additional_description: str, current_arg: str, criticism: str, **kwargs) -> str:
    return p.completion(
        Completion(
            deployment_id="p--G2s9okMTWWWh3d8YqLY2",
            llm_inputs={
                "additional_description": additional_description,
                "date": f"{datetime.now()}",
                "query": query,
                "current_arg": current_arg,
                "criticism": criticism,
            },
            **kwargs,
        )
    ).content


# This is the parent function which orchestrates the chaining. We'll define our trace_id and trace_name here
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


# let's return the full CompletionResponse to see what other information is returned
@trace
def refiner2(query: str, additional_description: str, current_arg: str, criticism: str, **kwargs) -> CompletionResponse:
    return p.completion(
        Completion(
            deployment_id="p--G2s9okMTvBEh3d8YqLY2",
            llm_inputs={
                "additional_description": additional_description,
                "date": f"{datetime.now()}",
                "query": query,
                "current_arg": current_arg,
                "criticism": criticism,
            },
            **kwargs,
        )
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
    result = argument_chain(
        "Whether moonshine is good for you.",
        additional_description="Provide a concise, few sentence argument on why moonshine is good for you.",
    )
    print(result)

    result2, trace_id = argument_chain2(
        "Whether moonshine is good for you.",
        additional_description="Provide a concise, few sentence argument on why moonshine is good for you.",
    )
    p.record_feedback(
        FeedbackRequest(
            trace_id=trace_id,
            score=0.7,  # 0.0 (bad) to 1.0 (good)
            target="Moonshine is wonderful. End of story.",
        )
    )
    print(result2)

    result3 = argument_chain3(
        "Whether moonshine is good for you.",
        additional_description="Provide a concise, few sentence argument on why moonshine is good for you.",
    )
    p.record_feedback(
        FeedbackRequest(
            trace_id=result3.trace_id,
            score=0.7,  # 0.0 (bad) to 1.0 (good)
            target="Moonshine is wonderful. End of story.",
        )
    )
    print(result3)
