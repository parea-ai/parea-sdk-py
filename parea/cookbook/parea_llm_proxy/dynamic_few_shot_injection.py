from typing import List, Optional

import os
import random
from datetime import datetime

from dotenv import load_dotenv
from pydantic import BaseModel

from parea import Parea, trace, trace_insert
from parea.schemas import Completion, CompletionResponse, FeedbackRequest, LLMInputs, Message, Role, TestCase

load_dotenv()

p = Parea(api_key=os.getenv("PAREA_API_KEY"))


class Person(BaseModel):
    name: str
    email: str


class Email(BaseModel):
    contact: Person
    email_sent: str


mock_DB: dict[str, Email] = {}
NUM_INTERACTIONS = 5


def call_llm(messages: List[Message]) -> CompletionResponse:
    return p.completion(Completion(llm_configuration=LLMInputs(model="gpt-4o", messages=messages)))


# Imitate collecting few shot examples from prod based on user feedback
@trace
def email_writer(main_objective: str, contact: Person, few_shot_examples: Optional[List[str]] = None) -> str:
    trace_insert({"end_user_identifier": contact.name, "metadata": {"has_few_shot_examples": bool(few_shot_examples)}})

    few_shot_examples_prompt = ("\nHere are some examples of good emails\n" + "\n".join(few_shot_examples)) if few_shot_examples else ""
    messages = [
        Message(
            role=Role.system,
            content=f"You are an AI who performs an email writing task based on the following objective: {main_objective}",
        ),
        # added date to prompt to avoid cache
        Message(
            role=Role.user,
            content=f"""
            Your email is from: {contact.model_dump()}
            Today's date is: {datetime.now().isoformat()}
            {few_shot_examples_prompt if few_shot_examples else ""}
            Email:
            """,
        ),
    ]
    response: CompletionResponse = call_llm(messages)
    trace_id = response.inference_id
    # insert into mock_DB
    mock_DB[trace_id] = Email(contact=contact, email_sent=response.content)
    return response.content


def mimic_prod(few_shot_limit: int = 3):
    contact = Person(name="John Doe", email="jdoe@email.com")
    dataset = p.get_collection("Good_Email_Examples")
    selected_few_shot_examples = None
    if dataset:
        testcases: list[TestCase] = list(dataset.test_cases.values())
        few_shot_examples = [case.inputs["email"] for case in testcases if case.inputs["user"] == contact.name]
        # This is simply taking most recent n examples. You can imagine adding additional logic to the dataset
        # that allows you to rank the examples based on some criteria
        selected_few_shot_examples = few_shot_examples[-few_shot_limit:] if few_shot_examples else None
    for interaction in range(NUM_INTERACTIONS):
        email = email_writer("Convincing email to gym to cancel membership early.", contact, selected_few_shot_examples)
        print(email)


def add_good_email_example_to_dataset(user_name, email):
    # Note: if the test case collection doesn't exist, we will create a new collection with the provided name and data
    p.add_test_cases([{"user": user_name, "email": email}], name="Good_Email_Examples")


def mimic_prod_feedback_collection():
    # imagine the trace_id of the email is stored in state in the UI, so when the user provides feedback, we can use it
    trace_ids = mock_DB.keys()
    for trace_id in trace_ids:
        score = random.uniform(0, 1)
        p.record_feedback(FeedbackRequest(trace_id=trace_id, score=score))
        # if the feedback is good, add it to the dataset to use later as a few-shot example
        if score >= 0.5:
            add_good_email_example_to_dataset(mock_DB[trace_id].contact.name, mock_DB[trace_id].email_sent)


if __name__ == "__main__":
    mimic_prod()
    mimic_prod_feedback_collection()
    # future llm calls will now have few-shot examples from the feedback collection
    mimic_prod()
    print("Done")
