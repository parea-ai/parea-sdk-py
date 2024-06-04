import os
import time

import openai
from dotenv import load_dotenv
from openai.pagination import SyncCursorPage
from openai.types.beta import Thread
from openai.types.beta.threads import Message, Run

from parea import Parea, trace

load_dotenv()

p = Parea(api_key=os.getenv("PAREA_API_KEY"))
client = openai.OpenAI()
p.wrap_openai_client(client)

QUESTIONS = ["I need to solve the equation `3x + 11 = 14`. Can you help me?", "Could you explain linear algebra to me?", "I don't like math. What can I do?"]


def pretty_print(messages):
    print("# Messages")
    for m in messages:
        print(f"{m.role}: {m.content[0].text.value}")
    print()


@trace
def create_assistant(instructions: str):
    return client.beta.assistants.create(
        name="Math Tutor",
        instructions=instructions,
        tools=[{"type": "code_interpreter"}],
        model="gpt-4-turbo-preview",
    )


@trace
def submit_message(assistant_id: str, thread_id: str, user_message: str) -> Run:
    client.beta.threads.messages.create(thread_id=thread_id, role="user", content=user_message)
    return client.beta.threads.runs.create(
        thread_id=thread_id,
        assistant_id=assistant_id,
    )


@trace
def get_response(thread_id: str) -> SyncCursorPage[Message]:
    return client.beta.threads.messages.list(thread_id=thread_id, order="asc")


@trace
def create_thread_and_run(assistant_id: str, user_input: str) -> (Thread, Run):
    thread = client.beta.threads.create()
    run = submit_message(assistant_id, thread.id, user_input)
    return thread, run


@trace
def wait_on_run(run: Run, thread: Thread) -> Run:
    while run.status == "queued" or run.status == "in_progress":
        run = client.beta.threads.runs.retrieve(
            thread_id=thread.id,
            run_id=run.id,
        )
        time.sleep(0.5)
    return run


@trace
def run_until_complete(assistant_id: str, run_instructions: str) -> SyncCursorPage[Message]:
    thread, run = create_thread_and_run(assistant_id, run_instructions)
    wait_on_run(run, thread)
    response = get_response(thread.id)
    pretty_print(response)
    return response


@trace
def main(assistant_instructions: str) -> SyncCursorPage[Message]:
    assistant = create_assistant(assistant_instructions)
    response = None
    for question in QUESTIONS:
        response = run_until_complete(assistant.id, question)
    return response


if __name__ == "__main__":
    main("You are a personal math tutor. Write and run code to answer math questions.")
