from typing import Callable, List, Optional

import re
from collections import Counter

from parea.evals.utils import call_openai, embed, safe_json_loads, sent_tokenize
from parea.schemas.models import Log


def precision_response_context_factory(context_field: Optional[str] = "context") -> Callable[[Log], float]:
    """Prop. of tokens in model generation which are also present in the retrieved context."""

    def precision_response_context(log: Log) -> float:
        """Prop. of tokens in model generation which are also present in the retrieved context."""
        context = log.inputs[context_field]

        provider = log.configuration.provider
        model = log.configuration.model

        if provider == "openai":
            import tiktoken

            encoding = tiktoken.encoding_for_model(model)
            context_tokens = encoding.encode(context)
            output_tokens = encoding.encode(log.output)
        else:
            raise NotImplementedError

        if len(context_tokens) == 0:
            return 1.0
        elif len(output_tokens) == 0:
            return 0.0

        common_tokens = Counter(context_tokens) & Counter(output_tokens)
        num_common = sum(common_tokens.values())
        return num_common / len(output_tokens)

    return precision_response_context


def llm_critique_faithfulness_factory(
    question_field: Optional[str] = "question",
    context_field: Optional[str] = "context",
    model: Optional[str] = "gpt-4",
) -> Callable[[Log], float]:
    """Quantifies how much the generated answer can be inferred from the retrieved context."""

    def llm_critique_faithfulness(log: Log) -> float:
        question = log.inputs[question_field]
        evidence = log.inputs[context_field]
        output = log.output
        response = call_openai(
            model=model,
            messages=[
                {"role": "system", "content": "You are CompareGPT, a machine to verify the groundedness of predictions. Answer with " "only yes/no."},
                {
                    "role": "user",
                    "content": f"You are given a question, the corresponding evidence and a prediction from a model. Compare "
                    f'the "Prediction" and the "Evidence" to determine whether all the information of the '
                    f"prediction in present in the evidence or can be inferred from the evidence. You must answer "
                    f'"no" if there are any specific details in the prediction that are not mentioned in the '
                    f"evidence or cannot be inferred from the evidence.\n\n"
                    f"Question: {question}\n\nPrediction: {output}\n\nEvidence: {evidence}\n\nCompareGPT response:",
                },
            ],
            temperature=0.0,
        )
        return float("yes" in response.lower())

    return llm_critique_faithfulness


def recall_response(log: Log) -> float:
    """Prop. of tokens in target/reference answer which are also in model generation."""
    target = log.target
    output = log.output

    provider = log.configuration.provider
    model = log.configuration.model

    if provider == "openai":
        import tiktoken

        encoding = tiktoken.encoding_for_model(model)
        target_tokens = encoding.encode(target)
        output_tokens = encoding.encode(output)
    else:
        raise NotImplementedError

    if len(target_tokens) == 0:
        return 1.0
    common_tokens = Counter(target_tokens) & Counter(output_tokens)
    num_common = sum(common_tokens.values())
    return num_common / len(target_tokens)


def llm_critique_correctness_factory(
    question_field: Optional[str] = "question",
    model: Optional[str] = "gpt-4",
) -> Callable[[Log], float]:
    """Quantifies how much the generated answer matches the ground truth / target."""

    def llm_critique_correctness(log: Log) -> float:
        question = log.inputs[question_field]
        output = log.output
        target = log.target
        response = call_openai(
            model=model,
            messages=[
                {"role": "system", "content": "You are CompareGPT, a machine to verify the groundedness of predictions. Answer with " "only yes/no."},
                {
                    "role": "user",
                    "content": f"""You are given a question, the corresponding ground-truth answer and a prediction from a model. Compare the "Ground-truth answer" and the "Prediction" to determine whether the prediction correctly answers the question. All information in the ground-truth answer must be present in the prediction, including numbers and dates. You must answer "no" if there are any specific details in the ground-truth answer that are not mentioned in the prediction. There should be no contradicting statements in the prediction. The prediction may contain extra information. If the prediction states something as a possibility, treat it as a definitive answer.

Question: {question}
Ground-truth answer: {target}
Prediction: {output}

CompareGPT response:""",
                },
            ],
            temperature=0.0,
        )
        return float("yes" in response.lower())

    return llm_critique_correctness


def ragas_context_relevancy_factory(question_field: str = "question", context_fields: List[str] = ["context"]) -> Callable[[Log], float]:
    """Quantifies how much the retrieved context relates to the query."""

    def ragas_context_relevancy(log: Log) -> float:
        """Quantifies how much the retrieved context relates to the query."""
        question = log.inputs[question_field]
        context = "\n".join(log.inputs[context_field] for context_field in context_fields)

        extracted_sentences = call_openai(
            model="gpt-3.5-turbo-16k",
            messages=[
                {
                    "role": "user",
                    "content": f"""\
Please extract relevant sentences from the provided context that is absolutely required answer the following question. If no relevant sentences are found, or if you believe the question cannot be answered from the given context, return the phrase "Insufficient Information".  While extracting candidate sentences you're not allowed to make any changes to sentences from given context.

question:{question}
context:\n{context}
candidate sentences:\n""",
                }
            ],
            temperature=0.0,
        ).strip()
        if extracted_sentences.lower() == "insufficient information":
            return 0.0
        else:
            n_extracted_sentences = len(sent_tokenize(extracted_sentences))
            n_context_sentences = len(sent_tokenize(context))
            return n_extracted_sentences / n_context_sentences

    return ragas_context_relevancy


def ragas_answer_context_faithfulness_factory(question_field: str = "question", context_fields: List[str] = ["context"]) -> Callable[[Log], float]:
    """Quantifies how much the generated answer can be inferred from the retrieved context."""

    def ragas_answer_context_faithfulness(log: Log) -> float:
        """Quantifies how much the generated answer can be inferred from the retrieved context."""
        question = log.inputs[question_field]
        context = "\n".join(log.inputs[context_field] for context_field in context_fields)
        output = log.output

        completion = call_openai(
            model="gpt-3.5-turbo-16k",
            messages=[
                {
                    "role": "user",
                    "content": f"""\
Given a question and answer, create one or more statements from each sentence in the given answer.
question: Who was  Albert Einstein and what is he best known for?
answer: He was a German-born theoretical physicist, widely acknowledged to be one of the greatest and most influential physicists of all time. He was best known for developing the theory of relativity, he also made important contributions to the development of the theory of quantum mechanics.
statements:\nAlbert Einstein was born in Germany.\nAlbert Einstein was best known for his theory of relativity.
question: Cadmium Chloride is slightly soluble in this chemical, it is also called what?
answer: alcohol
statements:\nCadmium Chloride is slightly soluble in alcohol.
question: Were Shahul and Jithin of the same nationality?
answer: They were from different countries.
statements:\nShahul and Jithin were from different countries.
question:{question}
answer: {output}
statements:\n""",
                }
            ],
            temperature=0.0,
        )
        statements = completion.strip().split("\n")
        statements_formatted = [f"{i+1}. {s.strip()}" for i, s in enumerate(statements)]

        verdicts = (
            call_openai(
                model="gpt-3.5-turbo-16k",
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

    return ragas_answer_context_faithfulness


def ragas_answer_relevancy_factory(question_field: str = "question", n_generations: int = 3) -> Callable[[Log], float]:
    """Quantifies how much the generated answer relates to the query."""
    try:
        import numpy as np
    except ImportError:
        raise ImportError("Please install numpy to use this metric.")

    def ragas_answer_relevancy(log: Log) -> float:
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
Question:
""",
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

    return ragas_answer_relevancy


def ragas_context_ranking_factory(question_field: str = "question", context_fields: List[str] = ["context"], ranking_measurement="average_precision") -> Callable[[Log], float]:
    """Quantifies if the retrieved context is ranked by their relevancy"""
    try:
        import numpy as np
    except ImportError:
        raise ImportError("Please install numpy to use this metric.")

    def ragas_context_ranking(log: Log) -> float:
        """Quantifies if the retrieved context is ranked by their relevancy"""
        question = log.inputs[question_field]
        contexts = [log.inputs[context_field] for context_field in context_fields]

        verifications = []
        for context in contexts:
            response = call_openai(
                model="gpt-3.5-turbo-16k",
                messages=[
                    {
                        "role": "user",
                        "content": f"""\
Verify if the information in the given context is useful in answering the question.

question: What are the health benefits of green tea?
context:
This article explores the rich history of tea cultivation in China, tracing its roots back to the ancient dynasties. It discusses how different regions have developed their unique tea varieties and brewing techniques. The article also delves into the cultural significance of tea in Chinese society and how it has become a symbol of hospitality and relaxation.
verification:
{{"reason":"The context, while informative about the history and cultural significance of tea in China, does not provide specific information about the health benefits of green tea. Thus, it is not useful for answering the question about health benefits.", "verdict":"No"}}

question: How does photosynthesis work in plants?
context:
Photosynthesis in plants is a complex process involving multiple steps. This paper details how chlorophyll within the chloroplasts absorbs sunlight, which then drives the chemical reaction converting carbon dioxide and water into glucose and oxygen. It explains the role of light and dark reactions and how ATP and NADPH are produced during these processes.
verification:
{{"reason":"This context is extremely relevant and useful for answering the question. It directly addresses the mechanisms of photosynthesis, explaining the key components and processes involved.", "verdict":"Yes"}}

question:{question}
context:
{context}
verification:""",
                    }
                ],
                temperature=0.0,
            )
            verifications.append(response)

        if ranking_measurement == "average_precision":
            response = [safe_json_loads(item) for item in verifications]
            response = [int("yes" in resp.get("verdict", " ").lower()) if resp.get("verdict") else np.nan for resp in response]
            denominator = sum(response) + 1e-10
            numerator = sum([(sum(response[: i + 1]) / (i + 1)) * response[i] for i in range(len(response))])
            return numerator / denominator
        else:
            raise NotImplementedError

    return ragas_context_ranking


def ragas_percent_target_supported_by_context_factory(question_field: str = "question", context_fields: List[str] = ["context"]) -> Callable[[Log], float]:
    """Quantifies how many sentences in the target/ground truth are supported by the retrieved context."""

    def ragas_percent_target_supported_by_context(log: Log) -> float:
        """Quantifies how many sentences in the target/ground truth are supported by the retrieved context."""
        question = log.inputs[question_field]
        context = "\n".join(log.inputs[context_field] for context_field in context_fields)
        target = log.target

        classification = call_openai(
            model="gpt-3.5-turbo-16k",
            messages=[
                {
                    "role": "user",
                    "content": f"""\
Given a context, and an answer, analyze each sentence in the answer and classify if the sentence can be attributed to the given context or not. Output json with reason.


question: What can you tell me about albert Albert Einstein?
context: Albert Einstein (14 March 1879 – 18 April 1955) was a German-born theoretical physicist,widely held to be one of the greatest and most influential scientists of all time. Best known for developing the theory of relativity, he also made important contributions to quantum mechanics, and was thus a central figure in the revolutionary reshaping of the scientific understanding of nature that modern physics accomplished in the first decades of the twentieth century. His mass–energy equivalence formula E = mc2, which arises from relativity theory, has been called "the world's most famous equation". He received the 1921 Nobel Prize in Physics "for his services to theoretical physics, and especially for his discovery of the law of the photoelectric effect", a pivotal step in the development of quantum theory. His work is also known for its influence on the philosophy of science. In a 1999 poll of 130 leading physicists worldwide by the British journal Physics World, Einstein was ranked the greatest physicist of all time. His intellectual achievements and originality have made Einstein synonymous with genius.
answer: Albert Einstein born in 14 March 1879 was  German-born theoretical physicist, widely held to be one of the greatest and most influential scientists of all time. He received the 1921 Nobel Prize in Physics "for his services to theoretical physics. He published 4 papers in 1905.  Einstein moved to Switzerland in 1895
classification:
[
    {{
        "statement_1":"Albert Einstein, born on 14 March 1879, was a German-born theoretical physicist, widely held to be one of the greatest and most influential scientists of all time.",
        "reason": "The date of birth of Einstein is mentioned clearly in the context.",
        "Attributed": "Yes"
    }},
    {{
        "statement_2":"He received the 1921 Nobel Prize in Physics 'for his services to theoretical physics.",
        "reason": "The exact sentence is present in the given context.",
        "Attributed": "Yes"
    }},
    {{
        "statement_3": "He published 4 papers in 1905.",
        "reason": "There is no mention about papers he wrote in the given context.",
        "Attributed": "No"
    }},
    {{
        "statement_4":"Einstein moved to Switzerland in 1895.",
        "reason": "There is no supporting evidence for this in the given context.",
        "Attributed": "No"
    }}
]

question: who won 2020 icc world cup?
context: Who won the 2022 ICC Men's T20 World Cup?
The 2022 ICC Men's T20 World Cup, held from October 16 to November 13, 2022, in Australia, was the eighth edition of the tournament. Originally scheduled for 2020, it was postponed due to the COVID-19 pandemic. England emerged victorious, defeating Pakistan by five wickets in the final to clinch their second ICC Men's T20 World Cup title.
answer: England
classification:
[
    {{
        "statement_1":"England won the 2022 ICC Men's T20 World Cup.",
        "reason": "From context it is clear that England defeated Pakistan to win the World Cup.",
         "Attributed": "Yes"
    }}
]

question: {question}
context: {context}
answer: {target}
classification:
""",
                }
            ],
            temperature=0.0,
        )
        pattern = "\[\s*\{.*?\}(\s*,\s*\{.*?\})*\s*\]"
        match = re.search(pattern, classification.replace("\n", ""))
        if match:
            response = eval(classification)
            numerator = sum(item.get("Attributed").lower() == "yes" for item in response)
            return numerator / len(response)
        else:
            return 0.0

    return ragas_percent_target_supported_by_context
