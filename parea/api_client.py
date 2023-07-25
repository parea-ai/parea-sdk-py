from typing import Any, Optional

import httpx


class HTTPClient:
    _instance = None
    base_url = "https://optimus-prompt-backend.vercel.app/api/parea/v1"
    api_key = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.sync_client = httpx.Client(base_url=cls.base_url, timeout=60 * 3.0)
            cls._instance.async_client = httpx.AsyncClient(base_url=cls.base_url, timeout=60 * 3.0)
        return cls._instance

    def set_api_key(self, api_key: str):
        self.api_key = api_key

    def request(
        self,
        method: str,
        endpoint: str,
        data: Optional[dict[str, Any]] = None,
        params: Optional[dict[str, Any]] = None,
        api_key: Optional[str] = None,
    ) -> httpx.Response:
        """
        Makes an HTTP request to the specified endpoint.
        """
        headers = {"x-api-key": self.api_key} if self.api_key else api_key
        response = self.sync_client.request(method, endpoint, json=data, headers=headers, params=params)
        response.raise_for_status()
        return response

    async def request_async(
        self,
        method: str,
        endpoint: str,
        data: Optional[dict[str, Any]] = None,
        params: Optional[dict[str, Any]] = None,
        api_key: Optional[str] = None,
    ) -> httpx.Response:
        """
        Makes an asynchronous HTTP request to the specified endpoint.
        """
        headers = {"x-api-key": self.api_key} if self.api_key else api_key
        response = await self.async_client.request(method, endpoint, json=data, headers=headers, params=params)
        response.raise_for_status()
        return response

    def close(self):
        """
        Closes the synchronous HTTP client.
        """
        self.sync_client.close()

    async def close_async(self):
        """
        Closes the asynchronous HTTP client.
        """
        await self.async_client.aclose()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close_async()
