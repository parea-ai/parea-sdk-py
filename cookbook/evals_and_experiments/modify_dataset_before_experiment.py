import os

from dotenv import load_dotenv
from openai import OpenAI

from parea import Parea, trace
from parea.evals.rag import context_query_relevancy_factory
from parea.schemas import TestCase

load_dotenv()

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
p = Parea(api_key=os.getenv("PAREA_API_KEY"))
p.wrap_openai_client(client)

context_query_relevancy = context_query_relevancy_factory(model="gpt-4o", context_fields=["context"])


@trace(eval_funcs=[context_query_relevancy])
def run_experiment(question: str, context: str) -> str:
    return (
        client.chat.completions.create(
            model="gpt-4o",
            temperature=0,
            messages=[{"role": "user", "content": f"Answer question using context. Context: {context}. Question: {question}"}],
        )
        .choices[0]
        .message.content
    )


# You can fetch a dataset directly and then modify it to meet our needs before passing it to p.experiment.
def rename_information_to_context(num_samples: int = 3):
    dataset = p.get_collection("Example_Dataset_Name")
    if dataset:
        testcases: list[TestCase] = list(dataset.test_cases.values())
        # Assume dataset looks like this:
        # [
        #     inputs={"information": "Some long document", "question": "What is X?"}, target="X is Y" ...
        # ]
        return [{"context": case.inputs["information"], "question": case.inputs["question"], "target": case.target} for case in testcases[:num_samples]]
    return []


def main():
    data = rename_information_to_context()
    experiment = p.experiment("My_Experiment_Name", func=run_experiment, data=data)
    experiment.run()


if __name__ == "__main__":
    main()
