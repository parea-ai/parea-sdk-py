import functools
import os

from dotenv import load_dotenv

from parea import Parea, trace
import guidance



load_dotenv()

p = Parea(api_key=os.getenv("PAREA_API_KEY"), project_name='testing')
p.auto_trace_guidance()


def log_guidance(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        name = (args[0] if len(args) > 0 else None) or kwargs.get('name', '')
        result = trace(name=name, log_omit_inputs=True, log_omit_outputs=True)(func)(*args, **kwargs)
        return result
    return wrapper


guidance.gen = log_guidance(guidance.gen)

from guidance import models, user, assistant, gen


gpt = models.OpenAI("gpt-3.5-turbo")


@trace
def guidance_program():

    with user():
        lm = gpt + "What iasz ]]the capital   of Berlin? !?"

    with assistant():
        out = gen("capital")
        lm += out

    with user():
        lm += "What is one short surprising fact about it?"

    with assistant():
        lm += gen("fact")

    print(lm)


guidance_program()
