import os

import marvin
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from parea import Parea

load_dotenv()

p = Parea(api_key=os.getenv("PAREA_API_KEY"))
p.auto_trace_openai_clients("marvin")


class Location(BaseModel):
    city: str
    state: str = Field(description="2-letter abbreviation")


result = marvin.cast("the big apple", Location)
print(result)
