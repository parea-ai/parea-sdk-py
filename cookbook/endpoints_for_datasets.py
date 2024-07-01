import os

from dotenv import load_dotenv

from parea import Parea

load_dotenv()

p = Parea(api_key=os.getenv("PAREA_API_KEY"))


data = [{"problem": "1+2", "target": 3, "tags": ["easy"]}, {"problem": "Solve the differential equation dy/dx = 3y.", "target": "y = c * e^(3x)", "tags": ["hard"]}]

# this will create a new dataset on Parea named "Math problems".
# The dataset will have one column named "problem", and two columns using the reserved names "target" and "tags".
# when using this dataset the expected prompt template should have a placeholder for the varible problem.
p.create_test_collection(data, name="Math problems")

new_data = [{"problem": "Evaluate the integral ∫x^2 dx from 0 to 3.", "target": 9, "tags": ["hard"]}]
# this will add the new test cases to the existing "Math problems" dataset.
# New test cases must have the same columns as the existing dataset.
p.add_test_cases(new_data, name="Math problems")
# Or if you can use the dataset ID instead of the name
p.add_test_cases(new_data, dataset_id=121)
