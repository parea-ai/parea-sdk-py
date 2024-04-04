from typing import List, Optional

from attr import asdict

from parea.cache.cache import Cache
from parea.schemas.models import CacheRequest, TraceLog
from parea.utils.universal_encoder import json_dumps


class InMemoryCache(Cache):
    def __init__(self):
        self.cache = {}
        self.logs = []

    def get(self, key: CacheRequest) -> Optional[TraceLog]:
        return self.cache.get(json_dumps(asdict(key)))

    async def aget(self, key: CacheRequest) -> Optional[TraceLog]:
        return self.get(key)

    def set(self, key: CacheRequest, value: TraceLog):
        self.cache[json_dumps(asdict(key))] = value

    async def aset(self, key: CacheRequest, value: TraceLog):
        self.set(key, value)

    def invalidate(self, key: CacheRequest):
        key = json_dumps(asdict(key))
        if key in self.cache:
            del self.cache[key]

    async def ainvalidate(self, key: CacheRequest):
        self.invalidate(key)

    def log(self, value: TraceLog):
        self.logs.append(value)

    def read_logs(self) -> List[TraceLog]:
        return self.logs.copy()
