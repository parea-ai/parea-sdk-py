from typing import Any, Optional

from parea.helpers import gen_random_name
from parea.schemas.models import CreateTestCase, CreateTestCaseCollection
from parea.utils.universal_encoder import json_dumps


def create_test_collection(data: list[dict[str, Any]], name: Optional[str] = None) -> CreateTestCaseCollection:
    """Create a test case collection from a dictionary of test cases.
    Args:
        data: list of key-value pairs where keys represent input names.
            Each item in the list represent a test case row.
            Target and Tags are reserved keys. There can only be one target and tags key per dict item.
            If target is present it will represent the target/expected response for the inputs.
            If tags are present they must be a list of json_serializable values.
        name: A unique name for the test collection. If not provided a random name will be generated.

    Returns: CreateTestCaseCollection
    """
    if not name:
        name = gen_random_name()

    column_names = list(set(k for row in data for k in row.keys() if k not in ["target", "tags"]))
    test_cases = create_test_cases(data)

    return CreateTestCaseCollection(name=name, column_names=column_names, test_cases=test_cases)


def create_test_cases(data: list[dict[str, Any]]) -> list[CreateTestCase]:
    """Create a list of test cases from a dictionary.
    Args:
        data: list of key-value pairs where keys represent input names.
            Each item in the list represent a test case row.
            Target and Tags are reserved keys. There can only be one target and tags key per dict item.
            If target is present it will represent the target/expected response for the inputs.
            If tags are present they must be a list of json_serializable values.

    Returns: list[CreateTestCase]
    """
    test_cases: list[CreateTestCase] = []
    for row in data:
        inputs: dict[str, str] = {}
        target: Optional[str] = None
        tags: list = []
        for k, v in row.items():
            if k == "target":
                if target is not None:
                    print("There can only be one target key per test case. Only the first target will be used.")
                target = json_dumps(v)
            elif k == "tags":
                if not isinstance(v, list):
                    raise ValueError("Tags must be a list of json serializable values.")
                if tags:
                    print("There can only be one tags key per test case. Only the first set of tags will be used.")
                tags = [tag if isinstance(tag, str) else json_dumps(tag) for tag in v]
            else:
                inputs[k] = v if isinstance(v, str) else json_dumps(v)
        test_cases.append(CreateTestCase(inputs=inputs, target=target, tags=tags))

    return test_cases
