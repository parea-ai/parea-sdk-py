from typing import Dict, List, Union

import json
import warnings

import openai
import pysbd
from openai import __version__ as openai_version

seg = pysbd.Segmenter(language="en", clean=False)


def sent_tokenize(text: str) -> List[str]:
    """Split into sentences"""
    sentences = seg.segment(text)
    assert isinstance(sentences, list)
    return sentences


def safe_json_loads(s) -> Dict:
    try:
        return json.loads(s)
    except ValueError as e:
        warnings.warn(f"Invalid json: {e}")

    return {}


def call_openai(messages, model, temperature=1.0, max_tokens=None, top_p=1.0, frequency_penalty=0.0, presence_penalty=0.0, n=1) -> Union[str, List[str]]:
    if openai_version.startswith("0."):
        completion = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            n=n,
        )
        if n == 1:
            return completion.choices[0].message["content"]
        else:
            return [c.message["content"] for c in completion.choices]
    else:
        completion = openai.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            n=n,
        )
        if n == 1:
            return completion.choices[0].message.content
        else:
            return [c.message.content for c in completion.choices]


def embed(model, input) -> List[float]:
    if openai_version.startswith("0."):
        return openai.Embedding.create(model=model, input=input, encoding_format="float").data[0]["embedding"]
    else:
        return openai.embeddings.create(model=model, input=input, encoding_format="float").data[0].embedding
