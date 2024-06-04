import os

from dotenv import load_dotenv

from parea import Parea
from parea.schemas import ListExperimentUUIDsFilters

load_dotenv()

p = Parea(api_key=os.getenv("PAREA_API_KEY"))

experiments = p.list_experiments(ListExperimentUUIDsFilters(experiment_name_filter="Greeting"))
print(f"Num. experiments: {len(experiments)}")
trace_logs = p.get_experiment_trace_logs(experiments[0].uuid)
print(f"Num. trace logs: {len(trace_logs)}")
print(f"Trace log: {trace_logs[0]}")
