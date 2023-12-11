from parea.evals.utils import call_openai, sent_tokenize
from parea.schemas.log import Log


def self_check(log: Log) -> float:
    """Measures how consistent is the output of a model under resampling the response."""
    if log.configuration is None or log.configuration.messages is None:
        return 0.0

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
