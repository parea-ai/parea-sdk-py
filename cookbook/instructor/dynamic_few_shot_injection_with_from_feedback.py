from typing import List

import os
import random
import uuid

import instructor
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field

from parea import Parea, get_current_trace_id, trace
from parea.schemas import FeedbackRequest, TestCase

load_dotenv()

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
p = Parea(api_key=os.getenv("PAREA_API_KEY"), project_name="example_implicit_feedback")
p.wrap_openai_client(client, "instructor")
client = instructor.from_openai(client)


class TraceData(BaseModel):
    trace_id: str
    input: str
    output: str


class Catalog(BaseModel):
    id: int
    name: str
    cost: float


CATALOG: List[Catalog] = [
    Catalog(id=1, name="Dog treats", cost=10.0),
    Catalog(id=2, name="Cat treats", cost=20.0),
    Catalog(id=3, name="Fish treats", cost=30.0),
    Catalog(id=4, name="Bird treats", cost=40.0),
    Catalog(id=5, name="Rabbit treats", cost=50.0),
]


def handle_checkout(catalog_item: Catalog):
    # some logic to handle the checkout
    pass


def get_signed_in_user() -> str:
    # imagine this is a function that returns the signed-in user id
    return str(uuid.uuid4())


# we will store all the recommendations made as product_id to TraceData
RECOMMENDATIONS: dict[int, TraceData] = {}


def get_recommendations(few_shot_limit: int = 3) -> List[dict[str, str]]:
    dataset = p.get_collection("Good_Recc_Examples")
    testcases: list[TestCase] = list(dataset.test_cases.values()) if dataset else []
    return [testcase.inputs for testcase in testcases[:few_shot_limit]]


# using instructor to generate a typed response
class Recc(BaseModel):
    product: Catalog
    reason: str = Field(
        ...,
        description="Reason response on why you think the user will like this product, phrased as if taking directly "
        "to the user using one the following tones [Salesy, Fact focused, Benefit focused, "
        "Feature focused, Adamant].",
    )


def lmm_product_recommendation_for_user(user_interest: str) -> Recc:
    messages = [
        {"role": "user", "content": f"Based on the users interests, recommend one product from our catalog: {CATALOG}. User interest: {user_interest}"},
    ]
    if recommendations := get_recommendations():
        messages.append(
            {
                "role": "user",
                "content": f"Here are some example recommendations from the past that the user enjoyed. "
                f"Your reasoning should follow this style. Recommendations: {recommendations}",
            }
        )
    return client.chat.completions.create(model="gpt-4o", messages=messages, response_model=Recc)


@trace(end_user_identifier=get_signed_in_user())
def llm_shopping(user_interest: str) -> Recc:
    recc = lmm_product_recommendation_for_user(user_interest)
    RECOMMENDATIONS[recc.product.id] = TraceData(trace_id=get_current_trace_id(), input=user_interest, output=recc.model_dump_json())
    return recc


def shopping_cart_checkout(catalog_item: Catalog) -> None:
    if catalog_item.id in RECOMMENDATIONS:
        trace_data = RECOMMENDATIONS[catalog_item.id]
        p.add_test_cases(data=[{"user_interests": trace_data.input, "recommendation": trace_data.output}], name="Good_Recc_Examples")
        p.record_feedback(FeedbackRequest(trace_id=trace_data.trace_id, score=1.0))
    handle_checkout(catalog_item)


if __name__ == "__main__":
    user_interests = ["dogs", "cats", "small dogs", "rabbits"]
    for interest in user_interests:
        response = llm_shopping(interest)
        print(response)
        print("Checkout")
        # simulate 50% change user follows model recommendation
        chosen_item = response.product if random.random() > 0.5 else random.choice(CATALOG)
        shopping_cart_checkout(chosen_item)
    print("Done")
