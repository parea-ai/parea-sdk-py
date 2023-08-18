from typing import Optional, Tuple

import os
from datetime import datetime

from dotenv import load_dotenv

from parea import Parea
from parea.client import gen_trace_id
from parea.schemas.models import Completion, CompletionResponse, FeedbackRequest

load_dotenv()

p = Parea(api_key=os.getenv("PAREA_API_KEY"))


# We pass the deployment_id and the required inputs to the completion function along with the trace_id
def argument_generator(query: str, additional_description: str = "", trace_id: Optional[str] = None, **kwargs) -> str:
    return p.completion(
        Completion(
            deployment_id="p-RG8d9sssc_0cctwfpb_n6",
            llm_inputs={
                "additional_description": additional_description,
                "date": f"{datetime.now()}",
                "query": query,
            },
            trace_id=trace_id,
            **kwargs,
        )
    ).content


def critic(argument: str, trace_id: Optional[str] = None, **kwargs) -> str:
    return p.completion(
        Completion(
            deployment_id="p-fXgZytVVVJjXD_71TDR4s",
            llm_inputs={"argument": argument},
            trace_id=trace_id,
            **kwargs,
        )
    ).content


def refiner(query: str, additional_description: str, current_arg: str, criticism: str, trace_id: Optional[str] = None, **kwargs) -> str:
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
            trace_id=trace_id,
            **kwargs,
        )
    ).content


# This is the parent function which orchestrates the chaining. We'll define our trace_id and trace_name here
def argument_chain(query: str, additional_description: str = "") -> str:
    trace_id, trace_name = gen_trace_id(), "argument_chain"
    argument = argument_generator(query, additional_description, trace_id, trace_name=trace_name)
    criticism = critic(argument, trace_id)
    return refiner(query, additional_description, argument, criticism, trace_id)


def argument_chain2(query: str, additional_description: str = "") -> tuple[str, str]:
    trace_id, trace_name = gen_trace_id(), "argument_chain"
    argument = argument_generator(query, additional_description, trace_id, trace_name=trace_name)
    criticism = critic(argument, trace_id)
    return refiner(query, additional_description, argument, criticism, trace_id), trace_id


# let's return the full CompletionResponse to see what other information is returned
def refiner2(query: str, additional_description: str, current_arg: str, criticism: str, trace_id: Optional[str] = None, **kwargs) -> CompletionResponse:
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
            trace_id=trace_id,
            **kwargs,
        )
    )


def argument_chain3(query: str, additional_description: str = "") -> CompletionResponse:
    trace_id, parent_trace_name = gen_trace_id(), "argument_chain"
    tags = ["tutorial"]
    metadata = {"githash": "e38f04c83"}
    argument = argument_generator(query, additional_description, trace_id, trace_name=parent_trace_name, tags=tags, metadata=metadata)
    criticism = critic(argument, trace_id)
    return refiner2(query, additional_description, argument, criticism, trace_id)


if __name__ == "__main__":
    result, trace_id = argument_chain2(
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
    print(result)
