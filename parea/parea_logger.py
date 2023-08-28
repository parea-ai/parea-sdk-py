from attrs import asdict, define, field

from parea.api_client import HTTPClient
from parea.schemas.models import TraceLog

LOG_ENDPOINT = "/trace_log"


@define
class PareaLogger:
    _client: HTTPClient = field(init=False)

    def set_client(self, client: HTTPClient) -> None:
        self._client = client

    def record_log(self, data: TraceLog) -> None:
        self._client.request(
            "POST",
            LOG_ENDPOINT,
            data=asdict(data),
        )

    async def arecord_log(self, data: TraceLog) -> None:
        await self._client.request_async(
            "POST",
            LOG_ENDPOINT,
            data=asdict(data),
        )


parea_logger = PareaLogger()
