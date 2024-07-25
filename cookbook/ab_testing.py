import os
import random
from typing import Tuple

from openai import OpenAI

from parea import trace, trace_insert, Parea, get_current_trace_id
from parea.schemas import FeedbackRequest

client = OpenAI()
# instantiate Parea client
p = Parea(api_key=os.getenv('PAREA_API_KEY'))
# wrap OpenAI client to trace calls
p.wrap_openai_client(client)


@trace
def generate_email(user: str) -> Tuple[str, str]:
    """Randomly chooses a prompt to perform an A/B test for generating email. Returns the email and the trace ID.
    The latter is used to tie-back the collected feedback from the user."""
    if random.random() < 0.5:
        trace_insert({'metadata': {'ab_test_0': 'variant_0'}})
        prompt = f'Generate a long email for {user}'
    else:
        trace_insert({'metadata': {'ab_test_0': 'variant_1'}})
        prompt = f'Generate a short email for {user}'

    email = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
    ).choices[0].message.content

    return email, get_current_trace_id()


def main():
    # generate email and get trace ID
    email, trace_id = generate_email('Max Mustermann')

    # log user feedback on email using trace ID
    p.record_feedback(
        FeedbackRequest(
            trace_id=trace_id,
            score=1.0,
        )
    )


if __name__ == '__main__':
    main()
