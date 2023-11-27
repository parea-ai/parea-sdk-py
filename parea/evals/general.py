import ast
import re
from typing import Callable

from parea.evals.utils import sent_tokenize, call_openai
from parea.schemas.models import Log

one_score_pattern = re.compile("\[\[(\d+\.?\d*)\]\]")
one_score_pattern_backup = re.compile("\[(\d+\.?\d*)\]")


def judge_llm_factory(model: str) -> Callable[[Log], float]:
    """Measures the generated response quality by using a LLM on a scale of 1 to 10."""
    def _eval_judge_llm(log: Log) -> float:
        question = log.inputs["question"]
        output = log.output
        rating_response = call_openai(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant."
                },
                {
                    "role": "user",
                    "content":
                        f"[Instruction]\nPlease act as an impartial judge and evaluate the quality of the response "
                        f"provided by an AI assistant to the user question displayed below. Your evaluation should "
                        f"consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and "
                        f"level of detail of the response. Begin your evaluation by providing a short explanation. "
                        f"Be as objective as possible. After providing your explanation, you must rate the response "
                        f"on a scale of 1 to 10 by strictly following this format: \"[[rating]]\", for example: "
                        f"\"Rating: [[5]]\".\n\n[Question]\n{question}\n\n[The Start of Assistant's Answer]"
                        f"\n{output}\n[The End of Assistant's Answer]"
                }
            ],
            temperature=0.0,
        )
        match = re.search(one_score_pattern, rating_response)
        if not match:
            match = re.search(one_score_pattern_backup, rating_response)

        if match:
            rating = ast.literal_eval(match.groups()[0])
        else:
            rating = 0

        return rating / 10.0

    return _eval_judge_llm


judge_llm_gpt4 = judge_llm_factory("gpt-4")

judge_llm_gpt3t = judge_llm_factory("gpt-3.5-turbo")


def self_check_gpt(log: Log) -> float:
    """Measures how consistent is the output of a model under resampling the response."""
    messages = [m.to_dict() for m in log.configuration.messages]

    n_sampled_outputs = 5
    sampled_outputs = []
    for _ in range(n_sampled_outputs):
        response = call_openai(
            messages=messages,
            model=log.configuration.model,
            temperature=1.0,
            max_tokens=log.configuration.model_params.max_length,
            top_p=log.configuration.model_params.top_p,
            frequency_penalty=log.configuration.model_params.frequency_penalty,
            presence_penalty=log.configuration.model_params.presence_penalty,
        )
        sampled_outputs.append(response)

    sentences = sent_tokenize(log.output)

    sentences_scores = []
    for sentence in sentences:
        scores = []
        for sampled_output in sampled_outputs:
            response = call_openai(
                messages=[
                    {
                        'role': 'user',
                        'content': f'''Context: {sampled_output}
Sentence: {sentence}
Is the sentence supported by the context above?
Answer Yes or No:'''
                    }
                ],
                model='gpt-3.5-turbo',
                temperature=0.0,
            )
            scores.append(float('yes' in response.lower()))
        sentences_scores.append(sum(scores) / len(scores))

    return sum(sentences_scores) / len(sentences_scores)


def lm_vs_lm_factuality_factory(examiner_model: str = 'gpt-3.5-turbo') -> Callable[[Log], float]:
    """Using an examining LLM, measures the factuality of a claim. Examining LLM asks follow-up questions to the other
    LLM until it reaches a conclusion."""
    def lm_vs_lm_factuality(log: Log) -> float:
        output = log.output
        messages_examinee = [m.to_dict() for m in log.configuration.messages]

        # ask examiner for follow-up questions
        setup_prompt = f'''Your goal is to try to verify the correctness of the following claim: {output}, based on the background information you will gather. To gather this, You will provide short questions whose purpose will be to verify the correctness of the claim, and I will reply to you with the answers to these. Hopefully, with the help of the background questions and their answers, you will be able to reach a conclusion as to whether the claim is correct or possibly incorrect. Please keep asking questions as long as you’re yet to be sure regarding the true veracity of the claim. Please start with the first questions.'''
        messages_examiner = [{"role": "user", "content": setup_prompt}]
        follow_up_questions = call_openai(
            model=examiner_model,
            messages=messages_examiner,
            temperature=0.0,
        )
        messages_examiner += [{"role": "assistant", "content": follow_up_questions}]

        follow_up_prompt = '''(i) Do you have any follow-up questions? Please answer with Yes or No.
    (ii) What are the follow-up questions?'''
        # ask examinee follow-up questions until they reach a conclusion
        while follow_up_questions is not None:
            messages_examinee += [{"role": "user", "content": follow_up_questions}]
            follow_up_answers = call_openai(
                model=log.configuration.model,
                messages=messages_examinee,
                temperature=log.configuration.model_params.temp,
                top_p=log.configuration.model_params.top_p,
                frequency_penalty=log.configuration.model_params.frequency_penalty,
                presence_penalty=log.configuration.model_params.presence_penalty,
                max_tokens=log.configuration.model_params.max_length,
            )
            messages_examiner += [
                {"role": "assistant", "content": follow_up_answers},
                {"role": "user", "content": follow_up_prompt},
            ]

            examiner_response = call_openai(
                model=examiner_model,
                messages=messages_examiner,
                temperature=0.0,
            )
            messages_examiner += [{"role": "assistant", "content": examiner_response}]
            if 'yes' in examiner_response.lower():
                follow_up_questions = examiner_response
                messages_examinee += [{"role": "assistant", "content": follow_up_answers}]
            else:
                follow_up_questions = None

        # ask examiner for their conclusion
        factuality_decision_prompt = '''Based on the interviewee’s answers to your questions, what is your conclusion regarding the correctness of the claim? Do you think it is correct or incorrect?'''
        messages_examiner += [{"role": "user", "content": factuality_decision_prompt}]
        examiner_response = call_openai(
            model=examiner_model,
            messages=messages_examiner,
            temperature=0.0,
        )
        return float('incorrect' not in examiner_response.lower())

    return lm_vs_lm_factuality
