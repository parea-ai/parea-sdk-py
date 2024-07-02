from typing import Dict, List

import json
import os
import time

import openai
from attr import asdict
from dotenv import load_dotenv

from parea import InMemoryCache, Parea, get_current_trace_id, trace, write_trace_logs_to_csv
from parea.evals import call_openai
from parea.evals.chat import goal_success_ratio_factory
from parea.schemas import Log

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")
openai.api_type = "openai"

use_cache = False  # by using the in memory cache, you don't need a Parea API key
cache = InMemoryCache() if use_cache else None
Parea(api_key=os.getenv("PAREA_API_KEY"), cache=cache)


def friendliness(log: Log) -> float:
    output = log.output
    response = call_openai(
        [
            {"role": "system", "content": "You evaluate the friendliness of the following response on a scale of 0 to 10. You must only return a number."},
            {"role": "assistant", "content": output},
        ],
        model="gpt-4",
    )
    try:
        return float(response) / 10.0
    except TypeError:
        return 0.0


def usefulness(log: Log) -> float:
    user_input = log.inputs["messages"][-1]["content"]
    output = log.output
    response = call_openai(
        [
            {"role": "system", "content": "You evaluate the usefulness of the response given the user input on a scale of 0 to 10. You must only return a number."},
            {"role": "assistant", "content": f'''User input: "{user_input}"\nAssistant response: "{output}"'''},
        ],
        model="gpt-4",
    )
    try:
        return float(response) / 10.0
    except TypeError:
        return 0.0


@trace(eval_funcs=[friendliness, usefulness])
def helpful_the_second_time(messages: List[Dict[str, str]]) -> str:
    helpful_response = call_openai(
        [
            {"role": "system", "content": "You are a friendly, and helpful assistant that helps people with their homework."},
        ]
        + messages,
        model="gpt-4",
    )

    has_user_asked_before_raw = call_openai(
        [
            {
                "role": "system",
                "content": "Assess if the user has asked the last question before or is asking again for more \
information on a previous topic. If so, respond ASKED_BEFORE. Otherwise, respond NOT_ASKED_BEFORE.",
            }
        ]
        + messages,
        model="gpt-4",
    )
    has_user_asked_before = has_user_asked_before_raw == "ASKED_BEFORE"

    if has_user_asked_before:
        messages.append({"role": "assistant", "content": helpful_response})
        return helpful_response
    else:
        unhelfpul_response = call_openai(
            [
                {
                    "role": "system",
                    "content": "Given the helpful response to the user input below, please provide a slightly unhelpful \
    response which makes the user ask again in case they didn't ask already again because of a previous unhelpful answer. \
    In case the user asked again, please provide a last response",
                },
            ]
            + messages
            + [{"role": "assistant", "content": helpful_response}],
            model="gpt-4",
        )
        messages.append({"role": "assistant", "content": unhelfpul_response})
        return unhelfpul_response


goal_success_ratio = goal_success_ratio_factory(use_output=True)


@trace(eval_funcs=[goal_success_ratio], access_output_of_func=lambda x: x[0])
def unhelpful_chat():
    print("\nWelcome to the somewhat helpful chat! Type 'exit' to end the session.")

    trace_id = get_current_trace_id()

    messages = []
    while True:
        user_input = input("\nYou: ")

        if user_input.lower() == "exit":
            print("Goodbye!")
            break

        messages.append({"role": "user", "content": user_input})
        print("Bot:", helpful_the_second_time(messages))

    return messages, trace_id


def main():
    _, trace_id = unhelpful_chat()

    if os.getenv("PAREA_API_KEY"):
        print(f"You can view the logs at: https://app.parea.ai/logs/detailed/{trace_id}")
    if use_cache:
        time.sleep(5)  # wait for local eval function to finish
        path_csv = f"trace_logs-{int(time.time())}.csv"
        trace_logs = cache.read_logs()
        write_trace_logs_to_csv(path_csv, trace_logs)
        print(f"\nCSV-file of traces: {path_csv}")
        parent_trace = None
        for trace_log in trace_logs:
            if trace_log.trace_id == trace_id:
                parent_trace = trace_log
                break
        if parent_trace:
            print(f"Overall score(s):\n{json.dumps(parent_trace.scores, default=asdict, indent=2)}")


if __name__ == "__main__":
    main()
