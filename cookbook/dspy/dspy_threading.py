import contextvars
import os
from concurrent.futures import ThreadPoolExecutor

import dspy
from dotenv import load_dotenv

from parea import Parea

load_dotenv()

p = Parea(api_key=os.getenv("PAREA_API_KEY"))
p.trace_dspy()

gpt3_turbo = dspy.OpenAI(model="gpt-3.5-turbo-1106", max_tokens=300)
dspy.configure(lm=gpt3_turbo)


class QASignature(dspy.Signature):
    question = dspy.InputField()
    answer = dspy.OutputField()


class EnsembleQA(dspy.Module):
    def __init__(self):
        super().__init__()
        self.step1 = dspy.ChainOfThought(QASignature)
        self.step2 = dspy.ChainOfThought(QASignature)

    def forward(self, question):
        with ThreadPoolExecutor(max_workers=2) as executor:
            context1 = contextvars.copy_context()
            future1 = executor.submit(context1.run, self.step1, question=question)
            context2 = contextvars.copy_context()
            future2 = executor.submit(context2.run, self.step2, question=question + "?")

        answer1 = future1.result()
        answer2 = future2.result()

        return dspy.Prediction(answer=f"{answer1}\n\n{answer2}")


qa = EnsembleQA()
response = qa("Who are you?")
print(response.answer)
