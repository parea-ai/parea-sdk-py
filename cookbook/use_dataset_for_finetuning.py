import os

from dotenv import load_dotenv
from parea import Parea

load_dotenv()

p = Parea(api_key=os.getenv("PAREA_API_KEY"))

dataset = p.get_collection(188)  # Replace DATASET_ID with the actual dataset ID

print(dataset.testcases[0])
# dataset.write_to_finetune_jsonl("finetune.jsonl")
# a = TestCase(
#     id=9730,
#     test_case_collection_id=188,
#     inputs={
#         "messages": '[\n  {\n    "created_at": "2024-08-05T21:02:40.439Z",\n    "user_name": "Customer",\n    "rendered_message": "Hi"\n  },\n  {\n    "created_at": "2024-08-05T21:02:40.857Z",\n    "user_name": "Bot",\n    "rendered_message": "_Tarek_ here from Rasayel ☺️ \\nI am here to support you with any questions you may have about Rasayel. 😃\\nWhich language do you feel more comfortable with?\\nنحن هنا لنقدم لك الدعم في أي استفسار قد يكون لديك حول رسايل. 😃\\nأي لغة تشعر بالراحة أكثر في التحدث بها؟"\n  },\n  {\n    "created_at": "2024-08-05T21:03:02.737Z",\n    "user_name": "Customer",\n    "rendered_message": "English"\n  },\n  {\n    "created_at": "2024-08-05T21:03:03.246Z",\n    "user_name": "Bot",\n    "rendered_message": "Great!\\nSo what brings you to us today? ☺️"\n  }\n]'
#     },
#     target="",
#     tags=[""],
# )
print(dataset.filter_testcases(id=9730))
print(dataset.filter_testcases(test_case_collection_id=188))
print(dataset.filter_testcases(target=""))
print(dataset.filter_testcases(tags=[""]))
