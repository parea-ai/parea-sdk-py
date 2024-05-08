from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

import json
from functools import wraps

from dspy import Example, Module, Prediction
from wrapt import wrap_object

from parea import trace
from parea.schemas import EvaluationResult, Log
from parea.utils.trace_integrations.wrapt_utils import CopyableFunctionWrapper
from parea.utils.universal_encoder import json_dumps

_DSPY_MODULE_NAME = "dspy"
_DSP_MODULE_NAME = "dsp"


class DSPyInstrumentor:

    def instrument(self) -> None:
        # Instrument LM (language model) calls
        from dsp.modules.lm import LM
        from dspy import Predict

        language_model_classes = LM.__subclasses__()
        for lm in language_model_classes:
            wrap_object(
                module=_DSP_MODULE_NAME,
                name=lm.__name__ + ".basic_request",
                factory=CopyableFunctionWrapper,
                args=(_GeneralDSPyWrapper("request"),),
            )

        # Predict is a concrete (non-abstract) class that may be invoked
        # directly, but DSPy also has subclasses of Predict that override the
        # forward method. We instrument both the forward methods of the base
        # class and all subclasses.
        wrap_object(
            module=_DSPY_MODULE_NAME,
            name="Predict.forward",
            factory=CopyableFunctionWrapper,
            args=(_PredictForwardWrapper(),),
        )

        predict_subclasses = Predict.__subclasses__()
        for predict_subclass in predict_subclasses:
            wrap_object(
                module=_DSPY_MODULE_NAME,
                name=predict_subclass.__name__ + ".forward",
                factory=CopyableFunctionWrapper,
                args=(_PredictForwardWrapper(),),
            )

        wrap_object(
            module=_DSPY_MODULE_NAME,
            name="Retrieve.__call__",
            factory=CopyableFunctionWrapper,
            args=(_GeneralDSPyWrapper("forward"),),
        )

        wrap_object(
            module=_DSPY_MODULE_NAME,
            # At this time, dspy.Module does not have an abstract forward
            # method, but assumes that user-defined subclasses implement the
            # forward method and invokes that method using __call__.
            name="Module.__call__",
            factory=CopyableFunctionWrapper,
            args=(_GeneralDSPyWrapper("forward"),),
        )

        # At this time, there is no common parent class for retriever models as
        # there is for language models. We instrument the retriever models on a
        # case-by-case basis.
        wrap_object(
            module=_DSP_MODULE_NAME,
            name="ColBERTv2.__call__",
            factory=CopyableFunctionWrapper,
            args=(_GeneralDSPyWrapper("__call__"),),
        )


class _GeneralDSPyWrapper:
    def __init__(self, method_name: str):
        self._method_name = method_name

    def __call__(
        self,
        wrapped: Callable[..., Any],
        instance: Any,
        args: Tuple[type, Any],
        kwargs: Mapping[str, Any],
    ) -> Any:
        span_name = instance.__class__.__name__ + "." + self._method_name
        return trace(name=span_name)(wrapped)(*args, **kwargs)


class _PredictForwardWrapper:
    """
    A wrapper for the Predict class to have a chain span for each prediction
    """

    def __call__(
        self,
        wrapped: Callable[..., Any],
        instance: Any,
        args: Tuple[type, Any],
        kwargs: Mapping[str, Any],
    ) -> Any:
        from dspy import Predict

        # At this time, subclasses of Predict override the base class' forward
        # method and invoke the parent class' forward method from within the
        # overridden method. The forward method for both Predict and its
        # subclasses have been instrumented. To avoid creating duplicate spans
        # for a single invocation, we don't create a span for the base class'
        # forward method if the instance belongs to a proper subclass of Predict
        # with an overridden forward method.
        is_instance_of_predict_subclass = isinstance(instance, Predict) and (cls := instance.__class__) is not Predict
        has_overridden_forward_method = getattr(cls, "forward", None) is not getattr(Predict, "forward", None)
        wrapped_method_is_base_class_forward_method = wrapped.__qualname__ == Predict.forward.__qualname__
        if is_instance_of_predict_subclass and has_overridden_forward_method and wrapped_method_is_base_class_forward_method:
            return wrapped(*args, **kwargs)
        else:
            return trace(name=_get_predict_span_name(instance))(wrapped)(*args, **kwargs)


def _get_predict_span_name(instance: Any) -> str:
    """
    Gets the name for the Predict span, which are the composition of a Predict
    class or subclass and a user-defined signature. An example name would be
    "Predict(UserDefinedSignature).forward".
    """
    class_name = str(instance.__class__.__name__)
    if (signature := getattr(instance, "signature", None)) and (signature_name := _get_signature_name(signature)):
        return f"{class_name}({signature_name}).forward"
    return f"{class_name}.forward"


def _get_signature_name(signature: Any) -> Optional[str]:
    """
    A best-effort attempt to get the name of a signature.
    """
    if (
        # At the time of this writing, the __name__ attribute on signatures does
        # not return the user-defined class name, but __qualname__ does.
        qual_name := getattr(signature, "__qualname__", None)
    ) is None:
        return None
    return str(qual_name.split(".")[-1])


def convert_dspy_eval_to_parea(dspy_eval):
    """Given a DSPy evaluation function, convert it to a Parea evaluation function."""

    @wraps(dspy_eval)
    def wrapper(log: Log):
        inputs_dict = log.inputs or {}
        target_dict = json.loads(log.target) if log.target else {}
        output_dict = json.loads(log.output) if log.output else {}
        ex = Example(**inputs_dict, **target_dict).with_inputs(*inputs_dict.keys())
        pred = Prediction(**output_dict)
        score = dspy_eval(ex, pred)
        return EvaluationResult(score=score, name=dspy_eval.__name__)

    return wrapper


def convert_dspy_examples_to_parea_dicts(examples: List[Example]) -> List[Dict]:
    """Convert a list of DSPy examples to a list of dictionaries suitable for Parea."""
    converted_examples = []
    for example in examples:
        _dict = {k: v for k, v in example.inputs().items()}
        _target = {k: v for k, v in example.labels().items()}
        if _target:
            _dict["target"] = json_dumps(_target)
        converted_examples.append(_dict)
    return converted_examples


def attach_evals_to_module(module: Module, eval_funcs: List) -> Callable:
    """Attach evaluation functions to a DSPy module to apply them async after execution of module."""

    @trace(name=module.__class__.__name__, eval_funcs=[convert_dspy_eval_to_parea(f) for f in eval_funcs])
    def inner(*args, **kwargs):
        return module(*args, **kwargs)

    return inner
