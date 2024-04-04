from typing import List, Optional

from abc import ABC

from parea.schemas.models import CacheRequest, TraceLog


class Cache(ABC):
    def get(self, key: CacheRequest) -> Optional[TraceLog]:
        """
        Get a normal response from the cache.

        Args:
            key (CacheRequest): The cache key.

        Returns:
            Optional[TraceLog]: The cached response, or None if the key was not found.

        # noqa: DAR202
        # noqa: DAR401
        """
        raise NotImplementedError

    async def aget(self, key: CacheRequest) -> Optional[TraceLog]:
        """
        Get a normal response from the cache.

        Args:
            key (CacheRequest): The cache key.

        Returns:
            Optional[TraceLog]: The cached response, or None if the key was not found.

        # noqa: DAR202
        # noqa: DAR401
        """
        raise NotImplementedError

    def set(self, key: CacheRequest, value: TraceLog):
        """
        Set a normal response in the cache.

        Args:
            key (CacheRequest): The cache key.
            value (TraceLog): The response to cache.

        # noqa: DAR401
        """
        raise NotImplementedError

    async def aset(self, key: CacheRequest, value: TraceLog):
        """
        Set a normal response in the cache.

        Args:
            key (CacheRequest): The cache key.
            value (TraceLog): The response to cache.

        # noqa: DAR401
        """
        raise NotImplementedError

    def invalidate(self, key: CacheRequest):
        """
        Invalidate a key in the cache.

        Args:
            key (CacheRequest): The cache key.

        # noqa: DAR401
        """
        raise NotImplementedError

    async def ainvalidate(self, key: CacheRequest):
        """
        Invalidate a key in the cache.

        Args:
            key (CacheRequest): The cache key.

        # noqa: DAR401
        """
        raise NotImplementedError

    def log(self, value: TraceLog):
        """
        Log a response in the cache.

        Args:
            value (TraceLog): The response to log.

        # noqa: DAR401
        """
        raise NotImplementedError

    def read_logs(self) -> List[TraceLog]:
        """
        Read all logs from the cache.

        Returns:
            List[TraceLog]: All logs in the cache.

        # noqa: DAR202
        # noqa: DAR401
        """
        raise NotImplementedError
