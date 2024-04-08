from typing import Callable, Optional

from parea.evals.utils import call_openai
from parea.schemas.log import Log


def lm_vs_lm_factuality_factory(
    examiner_model: str = "gpt-4",
    is_azure: Optional[bool] = False,
) -> Callable[[Log], float]:
    """
    This factory creates an evaluation function that measures the factuality of an LLM's response to a given question.
    It is based on the paper [LM vs LM: Detecting Factual Errors via Cross Examination](https://arxiv.org/abs/2305.13281) which proposes using
    another LLM to assess an LLM response's factuality. To do this, the examining LLM generates follow-up questions to the
    original response until it can confidently determine the factuality of the response.
    This method outperforms prompting techniques such as asking the original model, "Are you sure?" or instructing the
    model to say, "I don't know," if it is uncertain.

    Args:
        is_azure: Whether to use the Azure API. Defaults to False.
        examiner_model: The model which will examine the original model. Currently, only supports OpenAI chat models.

    Returns:
        Callable[[Log], float]: A function that takes a log as input and returns a score between 0 and 1 indicating
        the factuality of the response.
    """

    def lm_vs_lm_factuality(log: Log) -> float:
        output = log.output
        messages_examinee = [m.to_dict() for m in log.configuration.messages]

        # ask examiner for follow-up questions
        setup_prompt = f"""Your goal is to try to verify the correctness of the following claim: "{output}", based on the background information you will gather. To gather this, You will provide short questions whose purpose will be to verify the correctness of the claim, and I will reply to you with the answers to these. Hopefully, with the help of the background questions and their answers, you will be able to reach a conclusion as to whether the claim is correct or possibly incorrect. Please keep asking questions as long as you’re yet to be sure regarding the true veracity of the claim. Please start with the first questions."""
        messages_examiner = [{"role": "user", "content": setup_prompt}]
        follow_up_questions = call_openai(
            model=examiner_model,
            messages=messages_examiner,
            temperature=0.0,
            is_azure=is_azure,
        )
        messages_examiner += [{"role": "assistant", "content": follow_up_questions}]
        n_rounds_follow_up_questions = 1

        follow_up_prompt = """(i) Do you have any follow-up questions? Please answer with Yes or No.
    (ii) What are the follow-up questions?"""
        # ask examinee follow-up questions until they reach a conclusion
        while follow_up_questions is not None:
            messages_examinee += [{"role": "user", "content": follow_up_questions}]
            follow_up_answers = call_openai(
                model=examiner_model if is_azure else log.configuration.model,
                messages=messages_examinee,
                temperature=log.configuration.model_params.temp,
                top_p=log.configuration.model_params.top_p,
                frequency_penalty=log.configuration.model_params.frequency_penalty,
                presence_penalty=log.configuration.model_params.presence_penalty,
                max_tokens=log.configuration.model_params.max_length,
                response_format=log.configuration.model_params.response_format,
                is_azure=is_azure,
            )
            messages_examiner.append({"role": "assistant", "content": follow_up_answers})

            if n_rounds_follow_up_questions > 3:
                break
            else:
                messages_examiner.append({"role": "user", "content": follow_up_prompt})
                n_rounds_follow_up_questions += 1

            examiner_response = call_openai(
                model=examiner_model,
                messages=messages_examiner,
                temperature=0.0,
                is_azure=is_azure,
            )
            messages_examiner += [{"role": "assistant", "content": examiner_response}]
            if "yes" in examiner_response.lower():
                follow_up_questions = examiner_response
                messages_examinee += [{"role": "assistant", "content": follow_up_answers}]
            else:
                follow_up_questions = None

        # ask examiner for their conclusion
        factuality_decision_prompt = (
            """Based on the interviewee’s answers to your questions, what is your conclusion regarding the correctness of the claim? Do you think it is correct or incorrect?"""
        )
        messages_examiner += [{"role": "user", "content": factuality_decision_prompt}]
        examiner_response = call_openai(
            model=examiner_model,
            messages=messages_examiner,
            temperature=0.0,
            is_azure=is_azure,
        )
        return float("incorrect" not in examiner_response.lower())

    return lm_vs_lm_factuality


lm_vs_lm_factuality_gpt4 = lm_vs_lm_factuality_factory("gpt-4")
lm_vs_lm_factuality_gpt3t = lm_vs_lm_factuality_factory("gpt-3.5-turbo-16k")
