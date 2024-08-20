from typing import Dict, List

import os

from dotenv import load_dotenv

from parea import Parea
from parea.schemas import FilterOperator, QueryParams

load_dotenv()


project_name = "default"
p = Parea(api_key=os.getenv("PAREA_API_KEY"), project_name=project_name)


def fetch_trace_logs_as_jsonl() -> List[Dict]:
    page_size = 100
    query_params = QueryParams(
        project_name=project_name,
        filter_field="trace_name",
        filter_value="personalize_email_german",
        filter_operator=FilterOperator.EQUALS,
        page_size=page_size,
        status="success",
    )
    initial_fetch = p.get_trace_logs(query_params)
    fetched_trace_logs = initial_fetch.results
    for page in range(1, initial_fetch.total_pages):
        query_params.page = page
        fetched_trace_logs.extend(p.get_trace_logs(query_params).results)
    return [trace_log.convert_to_jsonl_row_for_finetuning() for trace_log in fetched_trace_logs]


jsonl_rows = fetch_trace_logs_as_jsonl()
