from typing import Any, AsyncIterable, Callable, Dict, List, Optional

import asyncio
import json
import os
import time
from functools import wraps
from importlib import metadata as importlib_metadata

import httpx
from dotenv import load_dotenv

load_dotenv()

MAX_RETRIES = 8
BACKOFF_FACTOR = 0.5


def retry_on_502(func: Callable[..., Any]) -> Callable[..., Any]:
    """
    A decorator to retry a function or coroutine on encountering a 502 error.
    Parameters:
        - func: The function or coroutine to be decorated.
    Returns:
        - A wrapper function that incorporates retry logic.
    """

    @wraps(func)
    async def async_wrapper(*args, **kwargs):
        for retry in range(MAX_RETRIES):
            try:
                return await func(*args, **kwargs)
            except httpx.HTTPError as e:
                if not _should_retry(e, retry):
                    raise
                await asyncio.sleep(BACKOFF_FACTOR * (2**retry))

    @wraps(func)
    def sync_wrapper(*args, **kwargs):
        for retry in range(MAX_RETRIES):
            try:
                return func(*args, **kwargs)
            except httpx.HTTPError as e:
                if not _should_retry(e, retry):
                    raise
                time.sleep(BACKOFF_FACTOR * (2**retry))

    def _should_retry(error, current_retry):
        """Determines if the function should retry on error."""
        is_502_error = isinstance(error, httpx.HTTPStatusError) and error.response.status_code == 502
        is_last_retry = current_retry == MAX_RETRIES - 1
        return not is_last_retry and (isinstance(error, (httpx.ConnectError, httpx.ReadError, httpx.RemoteProtocolError)) or is_502_error)

    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    else:
        return sync_wrapper


class HTTPClient:
    _instance = None
    base_url = os.getenv("PAREA_BASE_URL", "https://parea-ai-backend-us-9ac16cdbc7a7b006.onporter.run/api/parea/v1")
    api_key = None
    integrations: List[str] = []

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.sync_client = httpx.Client(base_url=cls.base_url, timeout=60 * 3.0)
            cls._instance.async_client = httpx.AsyncClient(base_url=cls.base_url, timeout=60 * 3.0)
        return cls._instance

    def set_api_key(self, api_key: str):
        self.api_key = api_key

    def add_integration(self, integration: str):
        if integration not in self.integrations:
            self.integrations.append(integration)

    def _get_headers(self, api_key: Optional[str] = None) -> Dict[str, str]:
        headers = {
            "x-api-key": self.api_key or api_key,
            "x-sdk-version": get_version(),
            "x-sdk-language": "python",
        }
        if self.integrations:
            headers["x-sdk-integrations"] = ",".join(self.integrations)
        return headers

    @retry_on_502
    def request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
        api_key: Optional[str] = None,
    ) -> httpx.Response:
        """
        Makes an HTTP request to the specified endpoint.
        """
        headers = self._get_headers(api_key=api_key)
        try:
            response = self.sync_client.request(method, endpoint, json=data, headers=headers, params=params)
            response.raise_for_status()
            return response
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 422:
                # update the error message to include the validation errors
                e.args = (f"{e.args[0]}: {e.response.json()}",)
            raise

    @retry_on_502
    async def request_async(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
        api_key: Optional[str] = None,
    ) -> httpx.Response:
        """
        Makes an asynchronous HTTP request to the specified endpoint.
        """
        headers = self._get_headers(api_key=api_key)
        try:
            response = await self.async_client.request(method, endpoint, json=data, headers=headers, params=params)
            response.raise_for_status()
            return response
        except httpx.HTTPStatusError as e:
            print(f"HTTP Error {e.response.status_code} for {e.request.url}: {e.response.text}")
            raise

    @retry_on_502
    def stream_request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
        api_key: Optional[str] = None,
        chunk_size: Optional[int] = None,
    ):
        """
        Makes a streaming HTTP request to the specified endpoint, yielding chunks of data.
        """
        headers = self._get_headers(api_key=api_key)
        try:
            with self.sync_client.stream(method, endpoint, json=data, headers=headers, params=params, timeout=None) as response:
                response.raise_for_status()
                for chunk in response.iter_bytes(chunk_size):
                    yield parse_event_data(chunk)
        except httpx.HTTPStatusError as e:
            print(f"HTTP Error {e.response.status_code} for {e.request.url}: {e.response.text}")
            raise

    @retry_on_502
    async def stream_request_async(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
        api_key: Optional[str] = None,
        chunk_size: Optional[int] = None,
    ) -> AsyncIterable[bytes]:
        """
        Makes an asynchronous streaming HTTP request to the specified endpoint, yielding chunks of data.
        """
        headers = self._get_headers(api_key=api_key)
        try:
            async with self.async_client.stream(method, endpoint, json=data, headers=headers, params=params, timeout=None) as response:
                response.raise_for_status()
                async for chunk in response.aiter_bytes(chunk_size):
                    yield parse_event_data(chunk)
        except httpx.HTTPStatusError as e:
            print(f"HTTP Error {e.response.status_code} for {e.request.url}: {e.response.text}")
            raise

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


def parse_event_data(byte_data):
    decoded_data = byte_data.decode("utf-8")

    try:
        json_str = [line for line in decoded_data.split("\r\n") if line.startswith("data:")][0]
        json_str = json_str.replace("data: ", "")
        if json_str.startswith("ID_START"):
            return ""

        data_dict = json.loads(json_str)
        return data_dict["chunk"]
    except Exception as e:
        print(f"Error parsing event data: {e}")
        return None


def get_version() -> str:
    try:
        return importlib_metadata.version("parea-ai")
    except importlib_metadata.PackageNotFoundError:  # pragma: no cover
        return "unknown"
