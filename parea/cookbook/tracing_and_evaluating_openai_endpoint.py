import os
import time
from typing import Dict, List

import openai
from dotenv import load_dotenv

from parea import init, RedisCache
from parea.helpers import write_trace_logs_to_csv
from parea.utils.trace_utils import trace, get_current_trace_id

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")


use_cache = True
cache = RedisCache() if use_cache else None
init(api_key=os.getenv("PAREA_API_KEY"), cache=cache)


def call_llm(data: list[dict], model: str = "gpt-3.5-turbo", temperature: float = 0.0) -> str:
    return openai.ChatCompletion.create(model=model, temperature=temperature, messages=data).choices[0].message["content"]


def friendliness(inputs: Dict, output: str, target: str = None) -> float:
    response = call_llm(
        [
            {
                "role": "system",
                "content": "You evaluate the friendliness of the following response on a scale of 0 to 10. You must only return a number."
            },
            {"role": "assistant", "content": output},
        ],
        model='gpt-4'
    )
    try:
        return float(response) / 10.0
    except TypeError:
        return 0.0


def usefulness(inputs: Dict, output: str, target: str = None) -> float:
    user_input = inputs['messages'][-1]["content"]
    response = call_llm(
        [
            {
                "role": "system",
                "content": "You evaluate the usefulness of the response given the user input on a scale of 0 to 10. You must only return a number."
            },
            {"role": "assistant", "content": f'''User input: "{user_input}"\nAssistant response: "{output}"'''}
        ],
        model='gpt-4'
    )
    try:
        return float(response) / 10.0
    except TypeError:
        return 0.0


@trace(eval_funcs=[friendliness, usefulness])
def helpful_the_second_time(messages: List[Dict[str, str]]) -> str:
    helpful_response = call_llm(
        [
            {
                "role": "system",
                "content": "You are a friendly, and helpful assistant that helps people with their homework."
            },

        ] + messages,
        model='gpt-4'
    )

    has_user_asked_before_raw = call_llm(
        [
            {
                "role": "system",
                "content": "Assess if the user has asked the last question before or is asking again for more \
information on a previous topic. If so, respond ASKED_BEFORE. Otherwise, respond NOT_ASKED_BEFORE."
            }
        ] + messages,
        model='gpt-4'
    )
    has_user_asked_before = has_user_asked_before_raw == "ASKED_BEFORE"

    if has_user_asked_before:
        messages.append({"role": "assistant", "content": helpful_response})
        return helpful_response
    else:
        unhelfpul_response = call_llm(
            [
                {
                    "role": "system",
                    "content": "Given the helpful response to the user input below, please provide a slightly unhelpful \
    response which makes the user ask again in case they didn't ask already again because of a previous unhelpful answer. \
    In case the user asked again, please provide a last response"
                },
            ] + messages + [{"role": "assistant", "content": helpful_response}],
            model='gpt-4'
        )
        messages.append({"role": "assistant", "content": unhelfpul_response})
        return unhelfpul_response


@trace
def unhelpful_chat():
    print("Welcome to the chat! Type 'exit' to end the session.")

    trace_id = get_current_trace_id()

    messages = []
    while True:
        user_input = input("\nYou: ")

        if user_input.lower() == 'exit':
            print("Goodbye!")
            break

        messages.append({"role": "user", "content": user_input})
        print("Bot:", helpful_the_second_time(messages))

    return messages, trace_id


def main():
    _ , trace_id = unhelpful_chat()

    time.sleep(0.2)

    if use_cache:
        path_csv = f"trace_logs-{int(time.time())}.csv"
        trace_logs = cache.read_logs()
        write_trace_logs_to_csv(path_csv, trace_logs)
        print(f"CSV-file of results: {path_csv}")
    if os.getenv("PAREA_API_KEY"):
        print(f'You can view the logs at: https://optimusprompt.ai/logs/detailed/{trace_id}')


if __name__ == "__main__":
    main()
