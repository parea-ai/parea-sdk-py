from typing import Callable, Optional

from parea.evals.utils import call_openai, embed
from parea.schemas.log import Log


def answer_relevancy_factory(
    question_field: str = "question",
    n_generations: int = 3,
    model: Optional[str] = "gpt-3.5-turbo-16k",
    embedding_model: str = "text-embedding-ada-002",
    is_azure: Optional[bool] = False,
) -> Callable[[Log], float]:
    """
    This factory creates an evaluation function that measures how relevant the generated response is to the given question.
    It is based on the paper [RAGAS: Automated Evaluation of Retrieval Augmented Generation](https://arxiv.org/abs/2309.15217)
    which suggests using an LLM to generate multiple questions that fit the generated answer and measure the cosine
    similarity of the generated questions with the original one.

    Args:
        is_azure: Whether to use the Azure API. Defaults to False.
        embedding_model: The model which should be used for embedding the text.
        model: The model which should be used for grading. Defaults to "gpt-3.5-turbo-16k".
        question_field: The key name/field used for the question/query of the user. Defaults to "question".
        n_generations: The number of questions which should be generated. Defaults to 3.

    Returns:
        Callable[[Log], float]: A function that takes a log as input and returns a score between 0 and 1 indicating
        if the generated response is relevant to the query.

    Raises:
        ImportError: If numpy is not installed.
    """
    try:
        import numpy as np
    except ImportError:
        raise ImportError("Please install numpy to use this metric.")

    def answer_relevancy(log: Log) -> float:
        """Quantifies how much the generated answer relates to the query."""
        question = log.inputs[question_field]
        output = log.output

        generated_questions = call_openai(
            model=model,
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
            is_azure=is_azure,
        )
        embedded_generated_questions = [embed(model=embedding_model, input=q, is_azure=is_azure) for q in generated_questions]
        embedded_question = embed(model=embedding_model, input=question, is_azure=is_azure)

        question_vec = np.asarray(embedded_question).reshape(1, -1)
        gen_question_vec = np.asarray(embedded_generated_questions)
        norm = np.linalg.norm(gen_question_vec, axis=1) * np.linalg.norm(question_vec, axis=1)
        return (np.dot(gen_question_vec, question_vec.T).reshape(-1) / norm).mean()

    return answer_relevancy
