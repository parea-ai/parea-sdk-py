from typing import Any, AsyncIterable, Dict, List, Optional

import json
import logging
import os
from importlib import metadata as importlib_metadata

import httpx
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential

load_dotenv()

logger = logging.getLogger()


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

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=4, max=10))
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
            logger.error(
                f"HTTP error {e.response.status_code} for {e.request.method} with: {e.args}",
                extra={"request_data": data, "request_params": params},
            )
            raise
        except httpx.TimeoutException as e:
            logger.error(
                f"Timeout error for {e.request.method} {e.request.url}",
                extra={"request_data": data, "request_params": params},
            )
            raise
        except httpx.RequestError as e:
            logger.error(
                f"Request error for {e.request.method} {e.request.url}: {str(e)}",
                extra={"request_data": data, "request_params": params},
            )
            raise

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=4, max=10))
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
            if e.response.status_code == 422:
                # update the error message to include the validation errors
                e.args = (f"{e.args[0]}: {e.response.json()}",)
            logger.error(
                f"HTTP error {e.response.status_code} for {e.request.method} with: {e.args}",
                extra={"request_data": data, "request_params": params},
            )
            raise
        except httpx.TimeoutException as e:
            logger.error(
                f"Timeout error for {e.request.method} {e.request.url}",
                extra={"request_data": data, "request_params": params},
            )
            raise
        except httpx.RequestError as e:
            logger.error(
                f"Request error for {e.request.method} {e.request.url}: {str(e)}",
                extra={"request_data": data, "request_params": params},
            )
            raise

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=4, max=10))
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
            if e.response.status_code == 422:
                # update the error message to include the validation errors
                e.args = (f"{e.args[0]}: {e.response.json()}",)
            logger.error(
                f"HTTP error {e.response.status_code} for {e.request.method} with: {e.args}",
                extra={"request_data": data, "request_params": params},
            )
            raise
        except httpx.TimeoutException as e:
            logger.error(
                f"Timeout error for {e.request.method} {e.request.url}",
                extra={"request_data": data, "request_params": params},
            )
            raise
        except httpx.RequestError as e:
            logger.error(
                f"Request error for {e.request.method} {e.request.url}: {str(e)}",
                extra={"request_data": data, "request_params": params},
            )
            raise

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=4, max=10))
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
            if e.response.status_code == 422:
                # update the error message to include the validation errors
                e.args = (f"{e.args[0]}: {e.response.json()}",)
            logger.error(
                f"HTTP error {e.response.status_code} for {e.request.method} with: {e.args}",
                extra={"request_data": data, "request_params": params},
            )
            raise
        except httpx.TimeoutException as e:
            logger.error(
                f"Timeout error for {e.request.method} {e.request.url}",
                extra={"request_data": data, "request_params": params},
            )
            raise
        except httpx.RequestError as e:
            logger.error(
                f"Request error for {e.request.method} {e.request.url}: {str(e)}",
                extra={"request_data": data, "request_params": params},
            )
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
