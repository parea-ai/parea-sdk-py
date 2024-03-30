from typing import Union

from parea.evals.utils import call_openai, sent_tokenize
from parea.schemas.log import Log


def self_check(log: Log) -> Union[float, None]:
    """
    Given that many API-based LLMs don't reliably give access to the log probabilities of the generated tokens, assessing
    the certainty of LLM predictions via perplexity isn't possible.
    The [SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection for Generative Large Language Models](https://arxiv.org/abs/2303.08896) paper
    suggests measuring the average factuality of every sentence in a generated response. They generate additional responses
    from the LLM at a high temperature and check how much every sentence in the original answer is supported by the other generations.
    The intuition behind this is that if the LLM knows a fact, it's more likely to sample it. The authors find that this
    works well in detecting non-factual and factual sentences and ranking passages in terms of factuality.
    The authors noted that correlation with human judgment doesn't increase after 4-6 additional
    generations when using `gpt-3.5-turbo` to evaluate biography generations.

    Args:
        log (Log): The log object to of the trace evaluate.

    Returns:
        float: A score between 0 and 1 indicating the factuality of the response.
    """
    if log.configuration is None or log.configuration.messages is None:
        return None

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
            response_format=log.configuration.model_params.response_format,
        )
        sampled_outputs.append(response)

    sentences = sent_tokenize(log.output)

    if len(sentences) == 0:
        return 0.0

    sentences_scores = []
    for sentence in sentences:
        scores = []
        for sampled_output in sampled_outputs:
            response = call_openai(
                messages=[
                    {
                        "role": "user",
                        "content": f"""Context: {sampled_output}
Sentence: {sentence}
Is the sentence supported by the context above?
Answer Yes or No:""",
                    }
                ],
                model="gpt-3.5-turbo",
                temperature=0.0,
            )
            scores.append(float("yes" in response.lower()))
        sentences_scores.append(sum(scores) / len(scores))

    return sum(sentences_scores) / len(sentences_scores)
