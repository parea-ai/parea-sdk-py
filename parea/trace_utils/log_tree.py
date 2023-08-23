from typing import Dict, List

import json
import logging

from attrs import asdict

from parea.schemas.models import Completion, PartialLog

logger = logging.getLogger()


def build_trees(db_entries: list[PartialLog]) -> list[PartialLog]:
    entries_by_id: dict[str, PartialLog] = {entry.inference_id: entry for entry in db_entries}
    all_child_ids = {child_id for entry in db_entries for child_id in entry.children}
    root_ids = set(entries_by_id.keys()) - all_child_ids

    def build_subtree(entry_id: str) -> PartialLog:
        entry: PartialLog = entries_by_id[entry_id]

        if entry.llm_inputs:
            for k, v in entry.llm_inputs.items():
                if isinstance(v, Completion):
                    entry.llm_inputs[k] = asdict(v)

        subtree = PartialLog(
            **{
                "inference_id": entry.inference_id,
                "name": entry.trace_name,
                "start_timestamp": entry.start_timestamp,
                "llm_inputs": entry.llm_inputs,
                "output": entry.output,
                "end_timestamp": entry.end_timestamp,
                "children": [build_subtree(child_id) for child_id in entry.children],
                "metadata": entry.metadata,
                "tags": entry.tags,
                "target": entry.target,
                "end_user_identifier": entry.end_user_identifier,
            }
        )

        return subtree

    return [build_subtree(root_id) for root_id in root_ids]


def output_trace_data(db_entries: list[PartialLog]):
    trees: list[PartialLog] = build_trees(db_entries)
    for i, tree in enumerate(trees, start=1):
        print(f"Tree {i}:")
        print(tree)
        print(json.dumps(asdict(tree), indent=2))
