# checkout the associated tutorial at https://docs.parea.ai//tutorials/running-ab-tests/llm-generated-emails

from typing import Tuple

import os
import random

from openai import OpenAI

from parea import Parea, get_current_trace_id, parea_logger, trace, trace_insert
from parea.schemas import EvaluationResult, UpdateLog

client = OpenAI()
# instantiate Parea client
p = Parea(api_key=os.getenv("PAREA_API_KEY"))
# wrap OpenAI client to trace calls
p.wrap_openai_client(client)


ab_test_name = "long-vs-short-emails"


@trace  # decorator to trace functions with Parea
def generate_email(user: str) -> Tuple[str, str, str]:
    # randomly choose to generate a long or short email
    if random.random() < 0.5:
        variant = "variant_0"
        prompt = f"Generate a long email for {user}"
    else:
        variant = "variant_1"
        prompt = f"Generate a short email for {user}"
    # tag the requests with the A/B test name & chosen variant
    trace_insert(
        {
            "metadata": {
                "ab_test_name": ab_test_name,
                f"ab_test_{ab_test_name}": variant,
            }
        }
    )

    email = (
        client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
        )
        .choices[0]
        .message.content
    )
    # need to return in addition to the email, the trace_id and the chosen variant
    return email, get_current_trace_id(), variant


def capture_feedback(feedback: float, trace_id: str, ab_test_variant: str, user_corrected_email: str = None) -> None:
    field_name_to_value_map = {
        "scores": [EvaluationResult(name=f"ab_test_{ab_test_variant}", score=feedback, reason="any additional user feedback on why it's good/bad")],
    }
    if user_corrected_email:
        field_name_to_value_map["target"] = user_corrected_email

    parea_logger.update_log(
        UpdateLog(
            trace_id=trace_id,
            field_name_to_value_map=field_name_to_value_map,
        )
    )


def main():
    # generate email and get trace ID
    email, trace_id, ab_test_variant = generate_email("Max Mustermann")

    # create a biased feedback for shorter emals
    if ab_test_variant == "variant_1":
        user_feedback = 0.0 if random.random() < 0.7 else 1.0
    else:
        user_feedback = 0.0 if random.random() < 0.3 else 1.0

    capture_feedback(user_feedback, trace_id, ab_test_variant, "Hi Max")


if __name__ == "__main__":
    main()
