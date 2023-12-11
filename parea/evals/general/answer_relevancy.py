from typing import Callable

from parea.evals.utils import call_openai, embed
from parea.schemas.log import Log


def answer_relevancy_factory(question_field: str = "question", n_generations: int = 3) -> Callable[[Log], float]:
    """Quantifies how much the generated answer relates to the query."""
    try:
        import numpy as np
    except ImportError:
        raise ImportError("Please install numpy to use this metric.")

    def answer_relevancy(log: Log) -> float:
        """Quantifies how much the generated answer relates to the query."""
        question = log.inputs[question_field]
        output = log.output

        generated_questions = call_openai(
            model="gpt-3.5-turbo-16k",
            messages=[
                {
                    "role": "user",
                    "content": f"""\
Generate question for the given answer.
Answer:\nThe PSLV-C56 mission is scheduled to be launched on Sunday, 30 July 2023 at 06:30 IST / 01:00 UTC. It will be launched from the Satish Dhawan Space Centre, Sriharikota, Andhra Pradesh, India
Question: When is the scheduled launch date and time for the PSLV-C56 mission, and where will it be launched from?

Answer: {output}
Question:""",
                }
            ],
            temperature=0.0,
            n=n_generations,
        )
        embedded_generated_questions = [embed(model="text-embedding-ada-002", input=q) for q in generated_questions]
        embedded_question = embed(model="text-embedding-ada-002", input=question)

        question_vec = np.asarray(embedded_question).reshape(1, -1)
        gen_question_vec = np.asarray(embedded_generated_questions)
        norm = np.linalg.norm(gen_question_vec, axis=1) * np.linalg.norm(question_vec, axis=1)
        return (np.dot(gen_question_vec, question_vec.T).reshape(-1) / norm).mean()

    return answer_relevancy
