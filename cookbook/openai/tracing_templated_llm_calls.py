import os

from dotenv import load_dotenv
from openai import OpenAI

from parea import Parea

load_dotenv()

client = OpenAI()
p = Parea(api_key=os.getenv("PAREA_API_KEY"))
p.wrap_openai_client(client)

response = client.chat.completions.create(
    model="gpt-4",
    messages=[
        {"role": "user", "content": "Make up {{number}} people."},
    ],
    template_inputs={"number": "three"},  # with Parea wrapper, we can specify template_inputs which will appear as inputs and are used to fill-in the templated messages
    metadata={"template_id": "make-up-people-v1"},  # via Parea wrapper, can associate request with any metadata
)
print(response.choices[0].message.content)
