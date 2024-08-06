import os

from dotenv import load_dotenv
from openai import OpenAI
from parea import Parea
from pydantic import BaseModel

load_dotenv()

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
p = Parea(api_key=os.getenv("PAREA_API_KEY"))
p.wrap_openai_client(client)


class CalendarEvent(BaseModel):
    name: str
    date: str
    participants: list[str]


completion = client.beta.chat.completions.parse(
    model="gpt-4o-2024-08-06",
    messages=[
        {"role": "system", "content": "Extract the event information."},
        {"role": "user", "content": "Alice and Bob are going to a science fair on Friday."},
    ],
    response_format=CalendarEvent,
)


if __name__ == "__main__":
    event = completion.choices[0].message.parsed
    print(type(event))
    print(event)


TraceLog(configuration=LLMInputs(model='gpt-4o-2024-08-06', provider='openai', model_params=ModelParams(temp=1.0, top_p=1.0, frequency_penalty=0.0, presence_penalty=0.0, max_length=None, response_format="<class '__main__.CalendarEvent'>", safe_prompt=None), messages=[{'role': 'system', 'content': 'Extract the event information.'}, {'role': 'user', 'content': 'Alice and Bob are going to a science fair on Friday.'}], history=None, functions=[], function_call=None), inputs=None, output='{"name":"Science Fair","date":"Friday","participants":["Alice","Bob"]}', target=None, latency=1.771622, time_to_first_token=None, input_tokens=32, output_tokens=17, total_tokens=49, cost=0.000415, scores=[], trace_id='e894b955-c844-49f5-8480-71b0841f10b5', parent_trace_id='e894b955-c844-49f5-8480-71b0841f10b5', root_trace_id='e894b955-c844-49f5-8480-71b0841f10b5', start_timestamp='2024-08-06T20:26:42.365024+00:00', organization_id=None, project_uuid=None, error=None, status='success', deployment_id=None, cache_hit=False, output_for_eval_metrics=None, evaluation_metric_names=[], apply_eval_frac=1.0, feedback_score=None, trace_name='llm-openai', children=['600b25b7-417a-4409-8f96-afe8dd8fe8cf'], end_timestamp='2024-08-06T20:26:44.136646+00:00', end_user_identifier=None, session_id=None, metadata=None, tags=None, experiment_uuid=None, images=[], comments=None, annotations=None, depth=0, execution_order=0)
D {'configuration': {'model': 'gpt-4o-2024-08-06', 'provider': 'openai', 'model_params': {'temp': 1.0, 'top_p': 1.0, 'frequency_penalty': 0.0, 'presence_penalty': 0.0, 'max_length': None, 'response_format': "<class '__main__.CalendarEvent'>", 'safe_prompt': None}, 'messages': [{'role': 'system', 'content': 'Extract the event information.'}, {'role': 'user', 'content': 'Alice and Bob are going to a science fair on Friday.'}], 'history': None, 'functions': [], 'function_call': None}, 'inputs': None, 'output': '{"name":"Science Fair","date":"Friday","participants":["Alice","Bob"]}', 'target': None, 'latency': 1.771622, 'time_to_first_token': None, 'input_tokens': 32, 'output_tokens': 17, 'total_tokens': 49, 'cost': 0.000415, 'scores': [], 'trace_id': 'e894b955-c844-49f5-8480-71b0841f10b5', 'parent_trace_id': 'e894b955-c844-49f5-8480-71b0841f10b5', 'root_trace_id': 'e894b955-c844-49f5-8480-71b0841f10b5', 'start_timestamp': '2024-08-06T20:26:42.365024+00:00', 'organization_id': None, 'project_uuid': '1c4dfe49-bf84-11ee-92b3-3a9b36099f82', 'error': None, 'status': 'success', 'deployment_id': None, 'cache_hit': False, 'output_for_eval_metrics': None, 'evaluation_metric_names': [], 'apply_eval_frac': 1.0, 'feedback_score': None, 'trace_name': 'llm-openai', 'children': ['600b25b7-417a-4409-8f96-afe8dd8fe8cf'], 'end_timestamp': '2024-08-06T20:26:44.136646+00:00', 'end_user_identifier': None, 'session_id': None, 'metadata': None, 'tags': None, 'experiment_uuid': None, 'images': [], 'comments': None, 'annotations': None, 'depth': 0, 'execution_order': 0}
