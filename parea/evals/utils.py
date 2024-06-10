from typing import Callable, List, Optional, Union

import json
import warnings

import openai
import pysbd
import tiktoken
from attrs import define
from openai import __version__ as openai_version

from parea.parea_logger import parea_logger
from parea.schemas import EvaluationResult
from parea.schemas.log import Log
from parea.utils.trace_utils import thread_ids_running_evals, trace_data

seg = pysbd.Segmenter(language="en", clean=False)


@define
class EvalFuncTuple:
    name: str
    func: Callable


def sent_tokenize(text: str) -> List[str]:
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


def call_openai(
    messages, model, temperature=1.0, max_tokens=None, top_p=1.0, frequency_penalty=0.0, presence_penalty=0.0, response_format=None, n=1, is_azure=False
) -> Union[str, List[str]]:
    openai.api_type = "openai"
    if is_azure:
        from openai.lib.azure import AzureOpenAI

        openai.api_type = "azure"

        completion = AzureOpenAI().chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            response_format=response_format,
            n=n,
        )
        if n == 1:
            return completion.choices[0].message.content
        else:
            return [c.message.content for c in completion.choices]
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
            response_format=response_format,
            n=n,
        )
        if n == 1:
            return completion.choices[0].message.content
        else:
            return [c.message.content for c in completion.choices]


def embed(model, input, is_azure=False) -> List[float]:
    openai.api_type = "openai"
    if is_azure:
        from openai.lib.azure import AzureOpenAI

        openai.api_type = "azure"
        return AzureOpenAI().embeddings.create(model=model, input=input, encoding_format="float").data[0].embedding
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


# note name is extra odd to make sure that skip_decorator_if_func_in_stack works in 99.9% of cases
def _make_evaluations(trace_id: str, log: Log, eval_funcs: List[EvalFuncTuple], verbose: bool = False, sync: bool = False):
    scores = []
    for eval in eval_funcs:
        try:
            result = eval.func(log)
        except Exception as e:
            print(f"Error occurred calling evaluation function '{eval.func.__name__}', {e}")
            continue
        if isinstance(result, EvaluationResult):
            scores.append(result)
        elif isinstance(result, list):
            scores.extend(result)
        elif result is not None:
            scores.append(EvaluationResult(name=eval.name, score=result))

    trace_data.get()[trace_id].scores = scores
    trace_data.get()[trace_id].target = log.target
    data_with_scores = trace_data.get()[trace_id]
    thread_ids_running_evals.get().remove(trace_id)
    parea_logger.default_log(data=data_with_scores)
    if verbose:
        print("###Eval Results###")
        for score in scores:
            print(score)
        print(f"View trace at: https://app.parea.ai/logs/detailed/{trace_id} \n")
    if sync:
        return scores


def run_evals_in_thread_and_log(trace_id: str, log: Log, eval_funcs: List[EvalFuncTuple], verbose: bool = False):
    import threading

    logging_thread = threading.Thread(
        target=_make_evaluations,
        kwargs={"trace_id": trace_id, "log": log, "eval_funcs": eval_funcs, "verbose": verbose},
    )
    logging_thread.start()


def run_evals_synchronous(trace_id: str, log: Log, eval_funcs: List[EvalFuncTuple], verbose: bool = False) -> List[EvaluationResult]:
    return _make_evaluations(trace_id, log, eval_funcs, verbose, True)


def get_tokens(model: str, text: str) -> List[int]:
    if not text:
        return []
    fallback_model = "cl100k_base"
    try:
        encoding = tiktoken.encoding_for_model(model or fallback_model)
    except KeyError:
        encoding = tiktoken.get_encoding(fallback_model)
    try:
        return encoding.encode(text)
    except Exception as e:
        print(f"Error encoding text: {e}")
        return []


def get_context(log: Log, context_fields: Optional[List[str]] = None, as_list: bool = False) -> str:
    if context_fields:
        context_list = [log.inputs[context_field] for context_field in context_fields]
        return context_list if as_list else "\n".join(context_list)
    else:
        context = log.output
        try:
            loaded_context = json.loads(log.output)
            if isinstance(loaded_context, list):
                return loaded_context if as_list else "\n".join(loaded_context)
        except json.JSONDecodeError:
            pass
        return [context] if as_list else context
