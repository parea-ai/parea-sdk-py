from typing import Optional

import json
import logging
import os

import redis
from attr import asdict

from parea.cache.cache import Cache
from parea.schemas.models import CacheRequest, TraceLog

logger = logging.getLogger()


class RedisLRUCache(Cache):
    """A Redis-based LRU cache for caching both normal and streaming responses."""

    def __init__(
        self, host: str = os.getenv("REDIS_HOST", "localhost"), port: int = int(os.getenv("REDIS_PORT", 6379)), password: str = os.getenv("REDIS_PASSWORT", None), ttl=3600 * 6
    ):
        """
        Initialize the cache.

        Args:
            ttl (int): The default TTL for cache entries, in seconds.
        """
        self.r = redis.Redis(
            host=host,
            port=port,
            password=password,
        )
        self.ttl = ttl

    def get(self, key: CacheRequest) -> Optional[TraceLog]:
        try:
            result = self.r.get(json.dumps(asdict(key)))
            if result is not None:
                return TraceLog(**json.loads(result))
        except redis.RedisError as e:
            logger.error(f"Error getting key {key} from cache: {e}")
        return None

    async def aget(self, key: CacheRequest) -> Optional[TraceLog]:
        return self.get(key)

    def set(self, key: CacheRequest, value: TraceLog):
        try:
            self.r.set(json.dumps(asdict(key)), json.dumps(asdict(value)), ex=self.ttl)
        except redis.RedisError as e:
            logger.error(f"Error setting key {key} in cache: {e}")

    async def aset(self, key: CacheRequest, value: TraceLog):
        self.set(key, value)

    def invalidate(self, key: CacheRequest):
        """
        Invalidate a key in the cache.

        Args:
            key (str): The cache key.
        """
        try:
            self.r.delete(json.dumps(asdict(key)))
        except redis.RedisError as e:
            logger.error(f"Error invalidating key {key} from cache: {e}")

    async def ainvalidate(self, key: CacheRequest):
        self.invalidate(key)
