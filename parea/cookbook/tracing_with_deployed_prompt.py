import os
import time
from datetime import datetime

from dotenv import load_dotenv

from parea import Parea
from parea.schemas.models import Completion, CompletionResponse, FeedbackRequest
from parea.utils.trace_utils import get_current_trace_id, trace

load_dotenv()

p = Parea(api_key=os.getenv("DEV_API_KEY"))


@trace
def deployed_argument_generator(query: str, additional_description: str = "") -> str:
    return p.completion(
        Completion(
            deployment_id="p-Ar-Oi14-nBxHUiradyql9",
            llm_inputs={
                "additional_description": additional_description,
                "date": f"{datetime.now()}",
                "query": query,
            },
        )
    ).content


@trace
def deployed_critic(argument: str) -> str:
    return p.completion(
        Completion(
            deployment_id="p-W2yPy93tAczYrxkipjli6",
            llm_inputs={"argument": argument},
        )
    ).content


@trace
def deployed_refiner(query: str, additional_description: str, current_arg: str, criticism: str) -> str:
    return p.completion(
        Completion(
            deployment_id="p-8Er1Xo0GDGF2xtpmMOpbn",
            llm_inputs={
                "additional_description": additional_description,
                "date": f"{datetime.now()}",
                "query": query,
                "current_arg": current_arg,
                "criticism": criticism,
            },
        )
    ).content


@trace
def deployed_refiner2(query: str, additional_description: str, current_arg: str, criticism: str) -> CompletionResponse:
    return p.completion(
        Completion(
            deployment_id="p-8Er1Xo0GDGF2xtpmMOpbn",
            llm_inputs={
                "additional_description": additional_description,
                "date": f"{datetime.now()}",
                "query": query,
                "current_arg": current_arg,
                "criticism": criticism,
            },
        )
    )


@trace
def deployed_argument_chain(query: str, additional_description: str = "") -> str:
    argument = deployed_argument_generator(query, additional_description)
    criticism = deployed_critic(argument)
    return deployed_refiner(query, additional_description, argument, criticism)


@trace
def deployed_argument_chain2(query: str, additional_description: str = "") -> tuple[str, str]:
    trace_id = get_current_trace_id()
    argument = deployed_argument_generator(query, additional_description)
    criticism = deployed_critic(argument)
    return deployed_refiner(query, additional_description, argument, criticism), trace_id


@trace(
    tags=["cookbook-example-deployed", "feedback_tracked-deployed"],
    metadata={"source": "python-sdk", "deployed": True},
)
def deployed_argument_chain3(query: str, additional_description: str = "") -> CompletionResponse:
    argument = deployed_argument_generator(query, additional_description)
    criticism = deployed_critic(argument)
    return deployed_refiner2(query, additional_description, argument, criticism)


if __name__ == "__main__":
    result1 = deployed_argument_chain(
        "Whether coffee is good for you.",
        additional_description="Provide a concise, few sentence argument on why coffee is good for you.",
    )
    print(result1)

    result2, trace_id2 = deployed_argument_chain2(
        "Whether wine is good for you.",
        additional_description="Provide a concise, few sentence argument on why wine is good for you.",
    )
    time.sleep(3)
    p.record_feedback(
        FeedbackRequest(
            trace_id=trace_id2,
            score=0.0,  # 0.0 (bad) to 1.0 (good)
            target="Moonshine is wonderful.",
        )
    )
    print(result2)

    result3 = deployed_argument_chain3(
        "Whether moonshine is good for you.",
        additional_description="Provide a concise, few sentence argument on why moonshine is good for you.",
    )
    time.sleep(3)
    p.record_feedback(
        FeedbackRequest(
            trace_id=result3.inference_id,
            score=0.7,  # 0.0 (bad) to 1.0 (good)
            target="Moonshine is wonderful. End of story.",
        )
    )
    print(result3.error or result3.content)
