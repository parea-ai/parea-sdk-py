from typing import Optional

import json
import logging
import os
import time
import uuid

import redis
from attr import asdict

from parea.cache.cache import Cache
from parea.schemas.models import CacheRequest, TraceLog
from parea.utils.universal_encoder import json_dumps

logger = logging.getLogger()


def is_uuid(value: str) -> bool:
    try:
        uuid.UUID(value)
    except ValueError:
        return False
    return True


class RedisCache(Cache):
    """A Redis-based cache for caching LLM responses."""

    def __init__(
        self,
        key_logs: str = os.getenv("_parea_redis_logs_key", f"trace-logs-{time.time()}"),
        host: str = os.getenv("REDIS_HOST", "localhost"),
        port: int = int(os.getenv("REDIS_PORT", 6379)),
        password: str = os.getenv("REDIS_PASSWORT", None),
        ttl=3600 * 6,
    ):
        """
        Initialize the cache.

        Args:
            key_logs (str): The Redis key for the logs.
            host (str): The Redis host.
            port (int): The Redis port.
            password (str): The Redis password.
            ttl (int): The default TTL for cache entries, in seconds.
        """
        self.r = redis.Redis(
            host=host,
            port=port,
            password=password,
        )
        self.ttl = ttl
        self.key_logs = key_logs

    def get(self, key: CacheRequest) -> Optional[TraceLog]:
        try:
            result = self.r.get(json_dumps(asdict(key)))
            if result is not None:
                return TraceLog(**json.loads(result))
        except redis.RedisError as e:
            logger.error(f"Error getting key {key} from cache: {e}")
        return None

    async def aget(self, key: CacheRequest) -> Optional[TraceLog]:
        return self.get(key)

    def set(self, key: CacheRequest, value: TraceLog):
        try:
            self.r.set(json_dumps(asdict(key)), json_dumps(asdict(value)), ex=self.ttl)
        except redis.RedisError as e:
            logger.error(f"Error setting key {key} in cache: {e}")

    async def aset(self, key: CacheRequest, value: TraceLog):
        self.set(key, value)

    def invalidate(self, key: CacheRequest):
        """
        Invalidate a key in the cache.

        Args:
            key (CacheRequest): The cache key.
        """
        try:
            self.r.delete(json_dumps(asdict(key)))
        except redis.RedisError as e:
            logger.error(f"Error invalidating key {key} from cache: {e}")

    async def ainvalidate(self, key: CacheRequest):
        self.invalidate(key)

    def log(self, value: TraceLog):
        try:
            prev_logs = self.r.hget(self.key_logs, value.trace_id)
            log_dict = asdict(value)
            if prev_logs:
                log_dict = {**json.loads(prev_logs), **log_dict}
            self.r.hset(self.key_logs, value.trace_id, json_dumps(log_dict))
        except redis.RedisError as e:
            logger.error(f"Error adding to list in cache: {e}")

    def read_logs(self) -> list[TraceLog]:
        try:
            trace_logs_raw = self.r.hgetall(self.key_logs)
            trace_logs = []
            for trace_log_raw in trace_logs_raw.values():
                trace_logs.append(TraceLog(**json.loads(trace_log_raw)))
            return trace_logs
        except redis.RedisError as e:
            logger.error(f"Error reading list from cache: {e}")
        return []
