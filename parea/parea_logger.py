from attrs import asdict, define, field

from parea.api_client import HTTPClient
from parea.cache.redis import RedisLRUCache
from parea.schemas.models import TraceLog

LOG_ENDPOINT = "/trace_log"


@define
class PareaLogger:
    _client: HTTPClient = field(init=False)
    _redis_lru_cache: RedisLRUCache = field(init=False)

    def set_client(self, client: HTTPClient) -> None:
        self._client = client

    def set_redis_lru_cache(self, cache: RedisLRUCache) -> None:
        self._redis_lru_cache = cache

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

    def write_log(self, data: TraceLog) -> None:
        self._redis_lru_cache.log(data)

    def default_log(self, data: TraceLog) -> None:
        if self._redis_lru_cache:
            self.write_log(data)
        if self._client:
            self.record_log(data)


parea_logger = PareaLogger()
