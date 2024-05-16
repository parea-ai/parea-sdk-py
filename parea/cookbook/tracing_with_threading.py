from typing import List

import contextvars
import os
from concurrent.futures import ThreadPoolExecutor

from dotenv import load_dotenv

from parea import Parea, trace

load_dotenv()

p = Parea(api_key=os.getenv("PAREA_API_KEY"))


@trace
def llm_call(question):
    return f"I can't answer that question: {question}"


@trace
def multiple_llm_calls(question, n_calls: int = 2) -> List[str]:
    answers = []
    with ThreadPoolExecutor(max_workers=2) as executor:
        for _ in range(n_calls):
            context = contextvars.copy_context()
            future = executor.submit(context.run, llm_call, question)
            answers.append(future.result())
    return answers


response = multiple_llm_calls("Who are you?")
print(response)
