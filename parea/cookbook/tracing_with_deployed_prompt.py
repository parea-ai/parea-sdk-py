import json
import os
from datetime import datetime

from attrs import asdict
from dotenv import load_dotenv

from parea import Parea, get_current_trace_id, trace
from parea.schemas import Completion, CompletionResponse, FeedbackRequest

load_dotenv()

p = Parea(api_key=os.getenv("PAREA_API_KEY"))


def deployed_argument_generator(query: str, additional_description: str = "") -> str:
    return p.completion(
        Completion(
            deployment_id="p-XOh3kp8B0nIE82WgioPnr",
            llm_inputs={
                "additional_description": additional_description,
                "date": f"{datetime.now()}",
                "query": query,
            },
        )
    ).content


def deployed_critic(argument: str) -> str:
    return p.completion(
        Completion(
            deployment_id="p-PSOwRyIPaQRq4xQW3MbpV",
            llm_inputs={"argument": argument},
        )
    ).content


def deployed_refiner(query: str, additional_description: str, current_arg: str, criticism: str) -> str:
    return p.completion(
        Completion(
            deployment_id="p-bJ3-UKh9-ixapZafaRBsj",
            llm_inputs={
                "additional_description": additional_description,
                "date": f"{datetime.now()}",
                "query": query,
                "argument": current_arg,
                "criticism": criticism,
            },
        )
    ).content


def deployed_refiner2(query: str, additional_description: str, current_arg: str, criticism: str) -> CompletionResponse:
    return p.completion(
        Completion(
            deployment_id="p-bJ3-UKh9-ixapZafaRBsj",
            llm_inputs={
                "additional_description": additional_description,
                "date": f"{datetime.now()}",
                "query": query,
                "argument": current_arg,
                "criticism": criticism,
            },
        )
    )


@trace
def deployed_argument_chain(query: str, additional_description: str = "") -> str:
    argument = deployed_argument_generator(query, additional_description)
    criticism = deployed_critic(argument)
    return deployed_refiner(query, additional_description, argument, criticism)


@trace(
    tags=["cookbook-example-deployed", "feedback_tracked-deployed"],
    metadata={"source": "python-sdk", "deployed": "True"},
)
def deployed_argument_chain_tags_metadata(query: str, additional_description: str = "") -> tuple[CompletionResponse, str]:
    trace_id = get_current_trace_id()  # get parent's trace_id
    argument = deployed_argument_generator(query, additional_description)
    criticism = deployed_critic(argument)
    return deployed_refiner2(query, additional_description, argument, criticism), trace_id


if __name__ == "__main__":
    result1 = deployed_argument_chain(
        "Whether coffee is good for you.",
        additional_description="Provide a concise, few sentence argument on why coffee is good for you.",
    )
    print(result1)

    result2, trace_id = deployed_argument_chain_tags_metadata(
        "Whether coffee is good for you.",
        additional_description="Provide a concise, few sentence argument on why coffee is good for you.",
    )
    print(json.dumps(asdict(result2), indent=2))
    p.record_feedback(
        FeedbackRequest(
            trace_id=trace_id,
            score=0.7,  # 0.0 (bad) to 1.0 (good)
            target="Coffee is wonderful. End of story.",
        )
    )
