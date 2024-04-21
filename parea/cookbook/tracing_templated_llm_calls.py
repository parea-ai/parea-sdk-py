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
        {"role": "user", "content": "Make up {{number}} people. Some {abc}: def"},
    ],
    template_inputs={"number": "three"},
)
print(response.choices[0].message.content)
