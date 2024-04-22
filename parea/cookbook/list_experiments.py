import os

from dotenv import load_dotenv

from parea import Parea
from parea.schemas import ListExperimentUUIDsFilters

load_dotenv()

p = Parea(api_key=os.getenv("PAREA_API_KEY"))

exp_uuids = p.list_experiment_uuids(ListExperimentUUIDsFilters(experiment_name_filter="Greeting"))
print(f"Num. experiments: {len(exp_uuids)}")
trace_logs = p.get_experiment_trace_logs(exp_uuids[0])
print(f"Num. trace logs: {len(trace_logs)}")
