from typing import Callable, Union

import json
import warnings

import openai
import pysbd
from attrs import define
from openai import __version__ as openai_version

from parea.parea_logger import parea_logger
from parea.schemas.log import Log
from parea.schemas.models import NamedEvaluationScore, UpdateLog

seg = pysbd.Segmenter(language="en", clean=False)


@define
class EvalFuncTuple:
    name: str
    func: Callable


def sent_tokenize(text: str) -> list[str]:
    """Split into sentences"""
    sentences = seg.segment(text)
    assert isinstance(sentences, list)
    return sentences


def safe_json_loads(s) -> dict:
    try:
        return json.loads(s)
    except ValueError as e:
        warnings.warn(f"Invalid json: {e}")

    return {}


def call_openai(messages, model, temperature=1.0, max_tokens=None, top_p=1.0, frequency_penalty=0.0, presence_penalty=0.0, n=1) -> Union[str, list[str]]:
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


def embed(model, input) -> list[float]:
    if openai_version.startswith("0."):
        return openai.Embedding.create(model=model, input=input, encoding_format="float").data[0]["embedding"]
    else:
        return openai.embeddings.create(model=model, input=input, encoding_format="float").data[0].embedding


def dcg(y_true, ranking):
    """Discounted cumulative gain (DCG) at rank k."""
    import numpy as np

    y_true = np.asarray(y_true)
    ranking = np.asarray(ranking)
    rel = y_true[ranking]
    gains = 2**rel - 1
    discounts = np.log2(np.arange(len(ranking)) + 2)
    return np.sum(gains / discounts)


def ndcg(y_true, ranking):
    """Normalized discounted cumulative gain (NDCG) at rank k"""
    import numpy as np

    k = len(ranking)
    best_ranking = np.argsort(y_true)[::-1]
    best = dcg(y_true, best_ranking[:k])
    return dcg(y_true, ranking) / best


def make_evaluations(trace_id: str, log: Log, eval_funcs: list[EvalFuncTuple], verbose: bool = False):
    scores = [NamedEvaluationScore(name=eval.name, score=eval.func(log)) for eval in eval_funcs]
    if verbose:
        print(f"###Eval Results###")
        for score in scores:
            print(score)
    parea_logger.update_log(data=UpdateLog(trace_id=trace_id, field_name_to_value_map={"scores": scores, "target": log.target}))
    print(f"View trace at: https://app.parea.ai/logs/detailed/{trace_id} \n")


def run_evals_in_thread_and_log(trace_id: str, log: Log, eval_funcs: list[EvalFuncTuple], verbose: bool = False):
    import threading

    logging_thread = threading.Thread(
        target=make_evaluations,
        kwargs={"trace_id": trace_id, "log": log, "eval_funcs": eval_funcs, "verbose": verbose},
    )
    logging_thread.start()
