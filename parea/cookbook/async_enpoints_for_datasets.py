import asyncio
import os

from dotenv import load_dotenv

from parea import Parea
from parea.schemas import TestCaseCollection, TestCase, UpdateTestCase

load_dotenv()

p = Parea(api_key=os.getenv("PAREA_API_KEY"))


data = [{"problem": "1+2", "target": 3, "tags": ["easy"]}, {"problem": "Solve the differential equation dy/dx = 3y.", "target": "y = c * e^(3x)", "tags": ["hard"]}]
new_data = [{"problem": "Evaluate the integral ∫x^2 dx from 0 to 3.", "target": 9, "tags": ["hard"]}]


async def update_test_case_example():
    dataset: TestCaseCollection = await p.aget_collection("math_problems_v3")
    test_cases: dict[int, TestCase] = dataset.test_cases
    for test_case_id, test_case in test_cases.items():
        if "easy" in test_case.tags:
            # updated inputs must match the same k/v pair as original test case
            await p.aupdate_test_case(
                dataset_id=dataset.id,
                test_case_id=test_case_id,
                update_request=UpdateTestCase(inputs={"problem": "Evaluate the integral ∫x^6 dx from 0 to 9."}, target="((1/7)x^7)+C", tags=["hard"]),
            )
            break


async def main():
    await p.acreate_test_collection(data, name="math_problems_v3")
    await p.aadd_test_cases(new_data, dataset_id=182)
    await update_test_case_example()


if __name__ == "__main__":
    asyncio.run(main())
