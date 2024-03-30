import os

from dotenv import load_dotenv
from guidance import assistant, gen, models, select, user

from parea import Parea, trace

load_dotenv()

p = Parea(api_key=os.getenv("PAREA_API_KEY"), project_name="testing")
p.auto_trace_openai_clients()


gpt = models.OpenAI("gpt-3.5-turbo")


@trace
def guidance_program():

    with user():
        lm = gpt + "What is the capital of Italy?"

    with assistant():
        out = gen("capital")
        lm += out

    with user():
        lm += "What is one short surprising fact about it?"

    with assistant():
        lm += gen("fact")

    print(lm)


guidance_program()
