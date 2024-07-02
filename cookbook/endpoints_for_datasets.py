import os

from dotenv import load_dotenv

from parea import Parea
from parea.schemas import TestCase, TestCaseCollection, UpdateTestCase

load_dotenv()

p = Parea(api_key=os.getenv("PAREA_API_KEY"))


data = [{"problem": "1+2", "target": 3, "tags": ["easy"]}, {"problem": "Solve the differential equation dy/dx = 3y.", "target": "y = c * e^(3x)", "tags": ["hard"]}]

# this will create a new dataset on Parea named "math_problems_v4".
# The dataset will have one column named "problem", and two columns using the reserved names "target" and "tags".
# when using this dataset the expected prompt template should have a placeholder for the varible problem.
p.create_test_collection(data, name="math_problems_v4")

new_data = [{"problem": "Evaluate the integral ∫x^2 dx from 0 to 3.", "target": 9, "tags": ["hard"]}]
# this will add the new test cases to the existing "math_problems_v4" dataset.
# New test cases must have the same columns as the existing dataset.
p.add_test_cases(new_data, name="math_problems_v4")
# Or if you can use the dataset ID instead of the name
# p.add_test_cases(new_data, dataset_id=121)


def update_test_case_example():
    dataset: TestCaseCollection = p.get_collection("math_problems_v4")
    test_cases: dict[int, TestCase] = dataset.test_cases
    for test_case_id, test_case in test_cases.items():
        if "easy" in test_case.tags:
            # updated inputs must match the same k/v pair as original test case
            p.update_test_case(
                dataset_id=dataset.id,
                test_case_id=test_case_id,
                update_request=UpdateTestCase(inputs={"problem": "Evaluate the integral ∫x^6 dx from 0 to 9."}, target="((1/7)x^7)+C", tags=["hard"]),
            )
            break


if __name__ == "__main__":
    update_test_case_example()
