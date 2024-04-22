import os

from dotenv import load_dotenv

from parea import Parea
from parea.schemas import ListExperimentUUIDsFilters

load_dotenv()

p = Parea(api_key=os.getenv("PAREA_API_KEY"))

exp_uuids = p.list_experiment_uuids(ListExperimentUUIDsFilters())
print(len(exp_uuids))
print(exp_uuids)
