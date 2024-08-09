import os

from dotenv import load_dotenv

from parea import Parea
from parea.schemas import FilterOperator, QueryParams

load_dotenv()

p = Parea(api_key=os.getenv("PAREA_API_KEY"))

paginated_resp = p.get_trace_logs(QueryParams(project_name="default", filter_field="trace_name", filter_operator=FilterOperator.LIKE, filter_value="llm"))
print(f"Num. LLM logs fetched: {len(paginated_resp.results)} | total LLM logs: {paginated_resp.total}")
