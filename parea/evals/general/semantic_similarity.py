from typing import Callable

import numpy as np

from parea.evals.utils import embed
from parea.schemas import Log


def semantic_similarity_factory(embd_model: str = "text-embedding-3-small") -> Callable[[Log], float]:
    def semantic_similarity(log: Log) -> float:
        """Calculates semantic similarity between output and target"""
        output = log.output
        target = log.target

        output_vector = embed(model=embd_model, input=output)
        target_vector = embed(model=embd_model, input=target)

        output_vector = np.array(output_vector)
        target_vector = np.array(target_vector)

        return (np.dot(output_vector, target_vector) / (np.linalg.norm(output_vector) * np.linalg.norm(target_vector)) + 1) / 2

    return semantic_similarity


semantic_similarity_oai_3_small = semantic_similarity_factory(embd_model="text-embedding-3-small")
semantic_similarity_oai_3_large = semantic_similarity_factory(embd_model="text-embedding-3-large")
semantic_similarity_oai_ada_002 = semantic_similarity_factory(embd_model="text-embedding-ada-002")
