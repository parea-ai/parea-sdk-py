from typing import Callable, List, Optional

from parea.evals.utils import call_openai
from parea.schemas.log import Log


def answer_context_faithfulness_statement_level_factory(
    question_field: str = "question", context_fields: List[str] = ["context"], model: Optional[str] = "gpt-3.5-turbo-16k", is_azure: Optional[bool] = False
) -> Callable[[Log], float]:
    """
    This factory creates an evaluation function that measures the faithfulness of the generated answer to the given context
    by measuring how many statements from the generated answer can be inferred from the given context. It is based on the paper
    [RAGAS: Automated Evaluation of Retrieval Augmented Generation](https://arxiv.org/abs/2309.15217) which suggests using an LLM
    to create a list of all statements in the generated answer and assessing whether the given context supports each statement.

    Args:
        model: The model which should be used for grading. Defaults to "gpt-3.5-turbo-16k".
        is_azure: Whether to use the Azure API. Defaults to False.
        question_field: The key name/field used for the question/query of the user. Defaults to "question".
        context_fields: A list of key names/fields used for the retrieved contexts. Defaults to ["context"].

    Returns:
        Callable[[Log], float]: A function that takes a log as input and returns a score between 0 and 1 indicating
        if the retrieved context is relevant to the query.
    """

    def answer_context_faithfulness_statement_level(log: Log) -> float:
        """Quantifies how much the generated answer can be inferred from the retrieved context."""
        question = log.inputs[question_field]
        context = "\n".join(log.inputs[context_field] for context_field in context_fields)
        output = log.output

        completion = call_openai(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": f"""\
Given a question and answer, create one or more statements from each sentence in the given answer.
question: Who was  Albert Einstein and what is he best known for?
answer: He was a German-born theortical physicist, widely acknowledged to be one of the greatest and most influential physicists of all time. He was best known for developing the theory of relativity, he also made important contributions to the development of the theory of quantum mechanics.
statements:\nAlbert Einstein was born in Germany.\nAlbert Einstein was best known for his theory of relativity.
question: Cadmium Chloride is slightly soluble in this chemical, it is also called what?
answer: alcohol
statements:\nCadmium Chloride is slightly soluble in alcohol.
question: Were Shahul and Jithin of thee same nationality?
answer: They were from different countries.
statements:\nShahul and Jithin were from different countries.
question:{question}
answer: {output}
statements:\n""",
                }
            ],
            temperature=0.0,
            is_azure=is_azure,
        )
        statements = completion.strip().split("\n")
        statements_formatted = [f"{i+1}. {s.strip()}" for i, s in enumerate(statements)]

        verdicts = (
            call_openai(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": f"""\
Prompt: Natural language inference
Consider the given context and following statements, then determine whether they are supported by the information present in the context.Provide a brief explanation for each statement before arriving at the verdict (Yes/No). Provide a final verdict for each statement in order at the end in the given format. Do not deviate from the specified format.

Context:\nJohn is a student at XYZ University. He is pursuing a degree in Computer Science. He is enrolled in several courses this semester, including Data Structures, Algorithms, and Database Management. John is a diligent student and spends a significant amount of time studying and completing assignments. He often stays late in the library to work on his projects.
statements:\n1. John is majoring in Biology.\n2. John is taking a course on Artificial Intelligence.\n3. John is a dedicated student.\n4. John has a part-time job.\n5. John is interested in computer programming.\n
Answer:
1. John is majoring in Biology.
Explanation: John's major is explicitly mentioned as Computer Science. There is no information suggesting he is majoring in Biology.  Verdict: No.
2. John is taking a course on Artificial Intelligence.
Explanation: The context mentions the courses John is currently enrolled in, and Artificial Intelligence is not mentioned. Therefore, it cannot be deduced that John is taking a course on AI. Verdict: No.
3. John is a dedicated student.
Explanation: The prompt states that he spends a significant amount of time studying and completing assignments. Additionally, it mentions that he often stays late in the library to work on his projects, which implies dedication. Verdict: Yes.
4. John has a part-time job.
Explanation: There is no information given in the context about John having a part-time job. Therefore, it cannot be deduced that John has a part-time job.  Verdict: No.
5. John is interested in computer programming.
Explanation: The context states that John is pursuing a degree in Computer Science, which implies an interest in computer programming. Verdict: Yes.
Final verdict for each statement in order: No. No. Yes. No. Yes.
context:\n{context}
statements:\n{statements_formatted}
Answer:
""",
                    }
                ],
                temperature=0.0,
                is_azure=is_azure,
            )
            .lower()
            .strip()
        )
        final_answer = "Final verdict for each statement in order:".lower()
        if final_answer in verdicts:
            verdicts = verdicts[verdicts.find(final_answer) + len(final_answer) :]
            yes_count = sum(0 if "yes" in answer else 1 for answer in verdicts.strip().split(".") if answer != "")
            return yes_count / len(statements_formatted)
        else:
            return max(0, output.count("verdict: no")) / len(statements_formatted)

    return answer_context_faithfulness_statement_level
