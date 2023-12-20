from typing import Union

from langchain_core.tracers import BaseTracer
from langchain_core.tracers.schemas import ChainRun, LLMRun, Run, ToolRun

from parea.parea_logger import parea_logger
from parea.schemas.log import TraceIntegrations


class PareaAILangchainTracer(BaseTracer):
    def _persist_run(self, run: Union[Run, LLMRun, ChainRun, ToolRun]) -> None:
        parea_logger.record_vendor_log(run.dict(), TraceIntegrations.LANGCHAIN)
