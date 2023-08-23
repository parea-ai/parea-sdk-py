from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union, cast

import json
import logging
import os
import platform
import socket
import subprocess
import weakref
from collections.abc import Iterator, Mapping
from datetime import datetime
from enum import Enum
from functools import lru_cache
from urllib.parse import urlsplit
from uuid import UUID

from attrs import asdict
from dotenv import load_dotenv
from requests import ConnectionError, HTTPError, Response, Session
from requests.adapters import HTTPAdapter
from urllib3.util import Retry

from parea.schemas.models import Completion, CompletionResponse

if TYPE_CHECKING:
    pass


load_dotenv()

logger = logging.getLogger(__name__)


class PareaAPIError(Exception):
    """An error occurred while communicating with the Parea API."""


class PareaUserError(Exception):
    """An error occurred while communicating with the Parea API."""


class PareaError(Exception):
    """An error occurred while communicating with the Parea API."""


class PareaConnectionError(Exception):
    """Couldn't connect to the Parea API."""


def xor_args(*arg_groups: tuple[str, ...]) -> Callable:
    """Validate specified keyword args are mutually exclusive."""

    def decorator(func: Callable) -> Callable:
        def wrapper(*args: Any, **kwargs: Any) -> Callable:
            """Validate exactly one arg in each group is not None."""
            counts = [sum(1 for arg in arg_group if kwargs.get(arg) is not None) for arg_group in arg_groups]
            invalid_groups = [i for i, count in enumerate(counts) if count != 1]
            if invalid_groups:
                invalid_group_names = [", ".join(arg_groups[i]) for i in invalid_groups]
                raise ValueError("Exactly one argument in each of the following" " groups must be defined:" f" {', '.join(invalid_group_names)}")
            return func(*args, **kwargs)

        return wrapper

    return decorator


def raise_for_status_with_text(response: Response) -> None:
    """Raise an error with the response text."""
    try:
        response.raise_for_status()
    except HTTPError as e:
        raise ValueError(response.text) from e


def get_enum_value(enum: Enum | str) -> str:
    """Get the value of a string enum."""
    if isinstance(enum, Enum):
        return enum.value
    return enum


@lru_cache
def get_runtime_environment() -> dict:
    """Get information about the environment."""
    # Lazy import to avoid circular imports
    from parea import version

    return {
        "sdk_version": version,
        "library": "langsmith",
        "platform": platform.platform(),
        "runtime": "python",
        "runtime_version": platform.python_version(),
        "langchain_version": get_langchain_environment(),
    }


def _get_message_type(message: Mapping[str, Any]) -> str:
    if not message:
        raise ValueError("Message is empty.")
    if "lc" in message:
        if "id" not in message:
            raise ValueError(f"Unexpected format for serialized message: {message}" " Message does not have an id.")
        return message["id"][-1].replace("Message", "").lower()
    else:
        if "type" not in message:
            raise ValueError(f"Unexpected format for stored message: {message}" " Message does not have a type.")
        return message["type"]


def _get_message_fields(message: Mapping[str, Any]) -> Mapping[str, Any]:
    if not message:
        raise ValueError("Message is empty.")
    if "lc" in message:
        if "kwargs" not in message:
            raise ValueError(f"Unexpected format for serialized message: {message}" " Message does not have kwargs.")
        return message["kwargs"]
    else:
        if "data" not in message:
            raise ValueError(f"Unexpected format for stored message: {message}" " Message does not have data.")
        return message["data"]


def _convert_message(message: Mapping[str, Any]) -> dict[str, Any]:
    """Extract message from a message object."""
    message_type = _get_message_type(message)
    message_data = _get_message_fields(message)
    return {"type": message_type, "data": message_data}


def get_messages_from_inputs(inputs: Mapping[str, Any]) -> list[dict[str, Any]]:
    if "messages" in inputs:
        return [_convert_message(message) for message in inputs["messages"]]
    if "message" in inputs:
        return [_convert_message(inputs["message"])]
    raise ValueError(f"Could not find message(s) in run with inputs {inputs}.")


def get_message_generation_from_outputs(outputs: Mapping[str, Any]) -> dict[str, Any]:
    if "generations" not in outputs:
        raise ValueError(f"No generations found in in run with output: {outputs}.")
    generations = outputs["generations"]
    if len(generations) != 1:
        raise ValueError("Chat examples expect exactly one generation." f" Found {len(generations)} generations: {generations}.")
    first_generation = generations[0]
    if "message" not in first_generation:
        raise ValueError(f"Unexpected format for generation: {first_generation}." " Generation does not have a message.")
    return _convert_message(first_generation["message"])


def get_prompt_from_inputs(inputs: Mapping[str, Any]) -> str:
    if "prompt" in inputs:
        return inputs["prompt"]
    if "prompts" in inputs:
        prompts = inputs["prompts"]
        if len(prompts) == 1:
            return prompts[0]
        raise ValueError(f"Multiple prompts in run with inputs {inputs}." " Please create example manually.")
    raise ValueError(f"Could not find prompt in run with inputs {inputs}.")


def get_llm_generation_from_outputs(outputs: Mapping[str, Any]) -> str:
    if "generations" not in outputs:
        raise ValueError(f"No generations found in in run with output: {outputs}.")
    generations = outputs["generations"]
    if len(generations) != 1:
        raise ValueError(f"Multiple generations in run: {generations}")
    first_generation = generations[0]
    if "text" not in first_generation:
        raise ValueError(f"No text in generation: {first_generation}")
    return first_generation["text"]


@lru_cache
def get_docker_compose_command() -> list[str]:
    """Get the correct docker compose command for this system."""
    try:
        subprocess.check_call(
            ["docker", "compose", "--version"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return ["docker", "compose"]
    except (subprocess.CalledProcessError, FileNotFoundError):
        try:
            subprocess.check_call(
                ["docker-compose", "--version"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return ["docker-compose"]
        except (subprocess.CalledProcessError, FileNotFoundError):
            raise ValueError(
                "Neither 'docker compose' nor 'docker-compose'"
                " commands are available. Please install the Docker"
                " server following the instructions for your operating"
                " system at https://docs.docker.com/engine/install/"
            )


@lru_cache
def get_langchain_environment() -> str | None:
    try:
        import parea  # type: ignore

        return parea.version
    except:  # noqa
        return None


@lru_cache
def get_docker_version() -> str | None:
    import subprocess

    try:
        docker_version = subprocess.check_output(["docker", "--version"]).decode("utf-8").strip()
    except FileNotFoundError:
        docker_version = "unknown"
    except:  # noqa
        return None
    return docker_version


@lru_cache
def get_docker_compose_version() -> str | None:
    try:
        docker_compose_version = subprocess.check_output(["docker-compose", "--version"]).decode("utf-8").strip()
    except FileNotFoundError:
        docker_compose_version = "unknown"
    except:  # noqa
        return None
    return docker_compose_version


@lru_cache
def _get_compose_command() -> list[str] | None:
    try:
        compose_command = get_docker_compose_command()
    except ValueError as e:
        compose_command = [f"NOT INSTALLED: {e}"]
    except:  # noqa
        return None
    return compose_command


@lru_cache
def get_docker_environment() -> dict:
    """Get information about the environment."""
    compose_command = _get_compose_command()
    return {
        "docker_version": get_docker_version(),
        "docker_compose_command": " ".join(compose_command) if compose_command is not None else None,
        "docker_compose_version": get_docker_compose_version(),
    }


def _is_localhost(url: str) -> bool:
    """Check if the URL is localhost.

    Parameters
    ----------
    url : str
        The URL to check.

    Returns
    -------
    bool
        True if the URL is localhost, False otherwise.
    """
    try:
        netloc = urlsplit(url).netloc.split(":")[0]
        ip = socket.gethostbyname(netloc)
        return ip == "127.0.0.1" or ip.startswith("0.0.0.0") or ip.startswith("::")
    except socket.gaierror:
        return False


ID_TYPE = Union[UUID, str]


def _default_retry_config() -> Retry:
    """Get the default retry configuration.

    Returns
    -------
    Retry
        The default retry configuration.
    """
    return Retry(
        total=3,
        allowed_methods=None,  # Retry on all methods
        status_forcelist=[502, 503, 504, 408, 425, 429],
        backoff_factor=0.5,
        # Sadly urllib3 1.x doesn't support backoff_jitter
        raise_on_redirect=False,
        raise_on_status=False,
    )


def _serialize_json(obj: Any) -> str:
    """Serialize an object to JSON.

    Parameters
    ----------
    obj : Any
        The object to serialize.

    Returns
    -------
    str
        The serialized JSON string.

    Raises
    ------
    TypeError
        If the object type is not serializable.
    """
    if isinstance(obj, datetime):
        return obj.isoformat()
    else:
        return str(obj)


def close_session(session: Session) -> None:
    """Close the session.

    Parameters
    ----------
    session : Session
        The session to close.
    """
    logger.debug("Closing Client.session")
    session.close()


def _validate_api_key_if_hosted(api_url: str, api_key: str | None) -> None:
    """Verify API key is provided if url not localhost.

    Parameters
    ----------
    api_url : str
        The API URL.
    api_key : str or None
        The API key.

    Raises
    ------
    LangSmithUserError
        If the API key is not provided when using the hosted service.
    """
    if not _is_localhost(api_url):
        if not api_key:
            raise PareaUserError("API key must be provided when using hosted LangSmith API")


def _get_api_key(api_key: str | None) -> str | None:
    api_key = api_key if api_key is not None else os.getenv("DEV_API_KEY")
    if api_key is None or not api_key.strip():
        return None
    return api_key.strip().strip('"').strip("'")


def _get_api_url(api_url: str | None, api_key: str | None) -> str:
    _api_url = api_url if api_url is not None else "http://localhost:8000/api/parea/v1"
    if not _api_url.strip():
        raise PareaUserError("Parea API URL cannot be empty")
    return _api_url.strip().strip('"').strip("'").rstrip("/")


class Client:
    """Client for interacting with the LangSmith API."""

    __slots__ = [
        "__weakref__",
        "api_url",
        "api_key",
        "retry_config",
        "timeout_ms",
        "session",
    ]

    def __init__(
        self,
        api_url: str | None = None,
        *,
        api_key: str | None = None,
        retry_config: Retry | None = None,
        timeout_ms: int | None = None,
    ) -> None:
        """Initialize a Client instance.

        Parameters
        ----------
        api_url : str or None, default=None
            URL for the LangSmith API. Defaults to the LANGCHAIN_ENDPOINT
            environment variable or http://localhost:1984 if not set.
        api_key : str or None, default=None
            API key for the LangSmith API. Defaults to the LANGCHAIN_API_KEY
            environment variable.
        retry_config : Retry or None, default=None
            Retry configuration for the HTTPAdapter.
        timeout_ms : int or None, default=None
            Timeout in milliseconds for the HTTPAdapter.

        Raises
        ------
        LangSmithUserError
            If the API key is not provided when using the hosted service.
        """
        self.api_key = _get_api_key(api_key)
        self.api_url = _get_api_url(api_url, self.api_key)
        _validate_api_key_if_hosted(self.api_url, self.api_key)
        self.retry_config = retry_config or _default_retry_config()
        self.timeout_ms = timeout_ms or 7000
        # Create a session and register a finalizer to close it
        self.session = Session()
        weakref.finalize(self, close_session, self.session)

        # Mount the HTTPAdapter with the retry configuration
        adapter = HTTPAdapter(max_retries=self.retry_config)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

    def _repr_html_(self) -> str:
        """Return an HTML representation of the instance with a link to the URL.

        Returns
        -------
        str
            The HTML representation of the instance.
        """
        link = self._host_url
        return f'<a href="{link}", target="_blank" rel="noopener">LangSmith Client</a>'

    def __repr__(self) -> str:
        """Return a string representation of the instance with a link to the URL.

        Returns
        -------
        str
            The string representation of the instance.
        """
        return f"Client (API URL: {self.api_url})"

    @property
    def _host_url(self) -> str:
        """The web host url."""
        if _is_localhost(self.api_url):
            link = "http://localhost"
        elif "dev" in self.api_url.split(".", maxsplit=1)[0]:
            link = "http://localhost"
        else:
            link = "https://optimus-prompt-backend.vercel.app"
        return link

    @property
    def _headers(self) -> dict[str, str]:
        """Get the headers for the API request.

        Returns
        -------
        Dict[str, str]
            The headers for the API request.
        """
        headers = {}
        if self.api_key:
            headers["x-api-key"] = self.api_key
        return headers

    def request_with_retries(
        self,
        request_method: str,
        url: str,
        request_kwargs: Mapping,
    ) -> Response:
        """Send a request with retries.

        Parameters
        ----------
        request_method : str
            The HTTP request method.
        url : str
            The URL to send the request to.
        request_kwargs : Mapping
            Additional request parameters.

        Returns
        -------
        Response
            The response object.

        Raises
        ------
        LangSmithAPIError
            If a server error occurs.
        LangSmithUserError
            If the request fails.
        LangSmithConnectionError
            If a connection error occurs.
        LangSmithError
            If the request fails.
        """
        try:
            response = self.session.request(request_method, url, stream=False, **request_kwargs)
            raise_for_status_with_text(response)
            return response
        except HTTPError as e:
            if response is not None and response.status_code == 500:
                raise PareaAPIError(f"Server error caused failure to {request_method} {url} in" f" Parea API. {e}")
            else:
                raise PareaUserError(f"Failed to {request_method} {url} in LangSmith API. {e}")
        except ConnectionError as e:
            raise PareaConnectionError(f"Connection error caused failure to {request_method} {url}" "  in Parea API. Please confirm your PAREA_ENDPOINT.") from e
        except Exception as e:
            raise PareaError(f"Failed to {request_method} {url} in Parea API. {e}") from e

    def _get_with_retries(self, path: str, params: dict[str, Any] | None = None) -> Response:
        """Send a GET request with retries.

        Parameters
        ----------
        path : str
            The path of the request URL.
        params : Dict[str, Any] or None, default=None
            The query parameters.

        Returns
        -------
        Response
            The response object.

        Raises
        ------
        LangSmithAPIError
            If a server error occurs.
        LangSmithUserError
            If the request fails.
        LangSmithConnectionError
            If a connection error occurs.
        LangSmithError
            If the request fails.
        """
        return self.request_with_retries(
            "get",
            f"{self.api_url}/{path}",
            request_kwargs={
                "params": params,
                "headers": self._headers,
                "timeout": self.timeout_ms / 1000,
            },
        )

    def _get_paginated_list(self, path: str, *, params: dict | None = None) -> Iterator[dict]:
        """Get a paginated list of items.

        Parameters
        ----------
        path : str
            The path of the request URL.
        params : dict or None, default=None
            The query parameters.

        Yields
        ------
        dict
            The items in the paginated list.
        """
        params_ = params.copy() if params else {}
        offset = params_.get("offset", 0)
        params_["limit"] = params_.get("limit", 100)
        while True:
            params_["offset"] = offset
            response = self._get_with_retries(path, params=params_)
            items = response.json()
            if not items:
                break
            yield from items
            if len(items) < params_["limit"]:
                # offset and limit isn't respected if we're
                # querying for specific values
                break
            offset += len(items)

    def create_run(
        self,
        name: str,
        inputs: dict[str, Any],
        run_type: str,
        *,
        execution_order: int | None = None,
        **kwargs: Any,
    ) -> None:
        """Persist a run to the LangSmith API.

        Parameters
        ----------
        name : str
            The name of the run.
        inputs : Dict[str, Any]
            The input values for the run.
        run_type : str
            The type of the run, such as tool, chain, llm, retriever,
            embedding, prompt, or parser.
        execution_order : int or None, default=None
            The execution order of the run.
        **kwargs : Any
            Additional keyword arguments.

        Raises
        ------
        LangSmithUserError
            If the API key is not provided when using the hosted service.
        """
        project_name = kwargs.pop(
            "project_name",
            kwargs.pop(
                "session_name",
                os.environ.get(
                    # TODO: Deprecate PAREA_SESSION
                    "PAREA_PROJECT",
                    os.environ.get("PAREA_SESSION", "default"),
                ),
            ),
        )
        run_create = {
            **kwargs,
            "session_name": project_name,
            "name": name,
            "inputs": inputs,
            "run_type": run_type,
            "execution_order": execution_order if execution_order is not None else 1,
        }
        run_extra = cast(dict, run_create.setdefault("extra", {}))
        runtime = run_extra.setdefault("runtime", {})
        runtime_env = get_runtime_environment()
        run_extra["runtime"] = {**runtime_env, **runtime}
        headers = {**self._headers, "Accept": "application/json"}
        self.request_with_retries(
            "post",
            f"{self.api_url}/runs",
            request_kwargs={
                "data": json.dumps(run_create, default=_serialize_json),
                "headers": headers,
                "timeout": self.timeout_ms / 1000,
            },
        )

    def completion(self, data: Completion) -> CompletionResponse:
        headers = {**self._headers, "Accept": "application/json"}
        r = self.request_with_retries(
            "post",
            f"{self.api_url}/completion",
            request_kwargs={
                "data": json.dumps(asdict(data), default=_serialize_json),
                "headers": headers,
                "timeout": self.timeout_ms / 1000,
            },
        )
        return CompletionResponse(**r.json())

    def update_run(
        self,
        run_id: ID_TYPE,
        **kwargs: Any,
    ) -> None:
        """Update a run in the LangSmith API.

        Parameters
        ----------
        run_id : str or UUID
            The ID of the run to update.
        **kwargs : Any
            Additional keyword arguments.
        """
        headers = {**self._headers, "Accept": "application/json"}
        self.request_with_retries(
            "patch",
            f"{self.api_url}/runs/{run_id}",
            request_kwargs={
                "data": json.dumps(kwargs, default=_serialize_json),
                "headers": headers,
                "timeout": self.timeout_ms / 1000,
            },
        )

    # def _load_child_runs(self, run: Run) -> Run:
    #     """Load child runs for a given run.
    #
    #     Parameters
    #     ----------
    #     run : Run
    #         The run to load child runs for.
    #
    #     Returns
    #     -------
    #     Run
    #         The run with loaded child runs.
    #
    #     Raises
    #     ------
    #     LangSmithError
    #         If a child run has no parent.
    #     """
    #     child_runs = self.list_runs(id=run.child_run_ids)
    #     treemap: DefaultDict[UUID, List[Run]] = defaultdict(list)
    #     runs: Dict[UUID, Run] = {}
    #     for child_run in sorted(child_runs, key=lambda r: r.execution_order):
    #         if child_run.parent_run_id is None:
    #             raise PareaError(f"Child run {child_run.id} has no parent")
    #         treemap[child_run.parent_run_id].append(child_run)
    #         runs[child_run.id] = child_run
    #     run.child_runs = treemap.pop(run.id, [])
    #     for run_id, children in treemap.items():
    #         runs[run_id].child_runs = children
    #     return run

    #
    # def read_run(self, run_id: ID_TYPE, load_child_runs: bool = False) -> Run:
    #     """Read a run from the LangSmith API.
    #
    #     Parameters
    #     ----------
    #     run_id : str or UUID
    #         The ID of the run to read.
    #     load_child_runs : bool, default=False
    #         Whether to load nested child runs.
    #
    #     Returns
    #     -------
    #     Run
    #         The run.
    #     """
    #     response = self._get_with_retries(f"/runs/{run_id}")
    #     run = Run(**response.json(), _host_url=self._host_url)
    #     if load_child_runs and run.child_run_ids:
    #         run = self._load_child_runs(run)
    #     return run
    #
    # def list_runs(
    #     self,
    #     *,
    #     project_id: Optional[ID_TYPE] = None,
    #     project_name: Optional[str] = None,
    #     run_type: Optional[str] = None,
    #     dataset_name: Optional[str] = None,
    #     dataset_id: Optional[ID_TYPE] = None,
    #     reference_example_id: Optional[ID_TYPE] = None,
    #     query: Optional[str] = None,
    #     filter: Optional[str] = None,
    #     execution_order: Optional[int] = None,
    #     parent_run_id: Optional[ID_TYPE] = None,
    #     start_time: Optional[datetime] = None,
    #     end_time: Optional[datetime] = None,
    #     error: Optional[bool] = None,
    #     run_ids: Optional[List[ID_TYPE]] = None,
    #     limit: Optional[int] = None,
    #     offset: Optional[int] = None,
    #     order_by: Optional[Sequence[str]] = None,
    #     **kwargs: Any,
    # ) -> Iterator[Run]:
    #     """List runs from the LangSmith API.
    #
    #     Parameters
    #     ----------
    #     project_id : UUID or None, default=None
    #         The ID of the project to filter by.
    #     project_name : str or None, default=None
    #         The name of the project to filter by.
    #     run_type : str or None, default=None
    #         The type of the runs to filter by.
    #     dataset_name : str or None, default=None
    #         The name of the dataset to filter by.
    #     dataset_id : UUID or None, default=None
    #         The ID of the dataset to filter by.
    #     reference_example_id : UUID or None, default=None
    #         The ID of the reference example to filter by.
    #     query : str or None, default=None
    #         The query string to filter by.
    #     filter : str or None, default=None
    #         The filter string to filter by.
    #     execution_order : int or None, default=None
    #         The execution order to filter by.
    #     parent_run_id : UUID or None, default=None
    #         The ID of the parent run to filter by.
    #     start_time : datetime or None, default=None
    #         The start time to filter by.
    #     end_time : datetime or None, default=None
    #         The end time to filter by.
    #     error : bool or None, default=None
    #         Whether to filter by error status.
    #     run_ids : List[str or UUID] or None, default=None
    #         The IDs of the runs to filter by.
    #     limit : int or None, default=None
    #         The maximum number of runs to return.
    #     offset : int or None, default=None
    #         The number of runs to skip.
    #     order_by : Sequence[str] or None, default=None
    #         The fields to order the runs by.
    #     **kwargs : Any
    #         Additional keyword arguments.
    #
    #     Yields
    #     ------
    #     Run
    #         The runs.
    #     """
    #     if project_name is not None:
    #         if project_id is not None:
    #             raise ValueError("Only one of project_id or project_name may be given")
    #         project_id = self.read_project(project_name=project_name).id
    #     if dataset_name is not None:
    #         if dataset_id is not None:
    #             raise ValueError("Only one of dataset_id or dataset_name may be given")
    #         dataset_id = self.read_dataset(dataset_name=dataset_name).id
    #     query_params: Dict[str, Any] = {
    #         "session": project_id,
    #         "run_type": run_type,
    #         **kwargs,
    #     }
    #     if reference_example_id is not None:
    #         query_params["reference_example"] = reference_example_id
    #     if dataset_id is not None:
    #         query_params["dataset"] = dataset_id
    #     if query is not None:
    #         query_params["query"] = query
    #     if filter is not None:
    #         query_params["filter"] = filter
    #     if execution_order is not None:
    #         query_params["execution_order"] = execution_order
    #     if parent_run_id is not None:
    #         query_params["parent_run"] = parent_run_id
    #     if start_time is not None:
    #         query_params["start_time"] = start_time.isoformat()
    #     if end_time is not None:
    #         query_params["end_time"] = end_time.isoformat()
    #     if error is not None:
    #         query_params["error"] = error
    #     if run_ids is not None:
    #         query_params["id"] = run_ids
    #     if limit is not None:
    #         query_params["limit"] = limit
    #     if offset is not None:
    #         query_params["offset"] = offset
    #     if order_by is not None:
    #         query_params["order"] = order_by
    #     yield from (Run(**run, _host_url=self._host_url) for run in self._get_paginated_list("/runs", params=query_params))
    #
    # def delete_run(self, run_id: ID_TYPE) -> None:
    #     """Delete a run from the LangSmith API.
    #
    #     Parameters
    #     ----------
    #     run_id : str or UUID
    #         The ID of the run to delete.
    #     """
    #     response = self.session.delete(
    #         f"{self.api_url}/runs/{run_id}",
    #         headers=self._headers,
    #     )
    #     raise_for_status_with_text(response)
    #
    # def share_run(self, run_id: ID_TYPE, *, share_id: Optional[ID_TYPE] = None) -> str:
    #     """Get a share link for a run."""
    #     data = {
    #         "run_id": str(run_id),
    #         "share_token": share_id or str(uuid4()),
    #     }
    #     response = self.session.put(
    #         f"{self.api_url}/runs/{run_id}/share",
    #         headers=self._headers,
    #         json=data,
    #     )
    #     raise_for_status_with_text(response)
    #     share_token = response.json()["share_token"]
    #     return f"{self._host_url}/public/{share_token}/r"
    #
    # def unshare_run(self, run_id: ID_TYPE) -> None:
    #     """Delete share link for a run."""
    #     response = self.session.delete(
    #         f"{self.api_url}/runs/{run_id}/share",
    #         headers=self._headers,
    #     )
    #     raise_for_status_with_text(response)
    #
    # def read_run_shared_link(self, run_id: ID_TYPE) -> Optional[str]:
    #     response = self.session.get(
    #         f"{self.api_url}/runs/{run_id}/share",
    #         headers=self._headers,
    #     )
    #     raise_for_status_with_text(response)
    #     result = response.json()
    #     if result is None or "share_token" not in result:
    #         return None
    #     return f"{self._host_url}/public/{result['share_token']}/r"
    #
    # def run_is_shared(self, run_id: ID_TYPE) -> bool:
    #     """Get share state for a run."""
    #     link = self.read_run_shared_link(run_id)
    #     return link is not None


# import logging
# import os
# import platform
# import socket
# import weakref
# from datetime import datetime
# from enum import Enum
# from functools import lru_cache
# from typing import Any, Callable, Dict, Mapping, Optional, Tuple, Union, AsyncIterator
# from typing import (
#     TYPE_CHECKING,
#     cast,
# )
# from urllib.parse import urlsplit
# from uuid import UUID
#
# import httpx
# from attrs import asdict
# from dotenv import load_dotenv
# from httpx import HTTPStatusError
#
# from parea.schemas.models import Completion, CompletionResponse
#
# if TYPE_CHECKING:
#     pass
#
# load_dotenv()
#
# logger = logging.getLogger(__name__)
#
#
# class PareaAPIError(Exception):
#     """An error occurred while communicating with the Parea API."""
#
#
# class PareaUserError(Exception):
#     """An error occurred while communicating with the Parea API."""
#
#
# class PareaError(Exception):
#     """An error occurred while communicating with the Parea API."""
#
#
# class PareaConnectionError(Exception):
#     """Couldn't connect to the Parea API."""
#
#
# def xor_args(*arg_groups: Tuple[str, ...]) -> Callable:
#     """Validate specified keyword args are mutually exclusive."""
#
#     def decorator(func: Callable) -> Callable:
#         def wrapper(*args: Any, **kwargs: Any) -> Callable:
#             """Validate exactly one arg in each group is not None."""
#             counts = [sum(1 for arg in arg_group if kwargs.get(arg) is not None) for arg_group in arg_groups]
#             invalid_groups = [i for i, count in enumerate(counts) if count != 1]
#             if invalid_groups:
#                 invalid_group_names = [", ".join(arg_groups[i]) for i in invalid_groups]
#                 raise ValueError("Exactly one argument in each of the following" " groups must be defined:" f" {', '.join(invalid_group_names)}")
#             return func(*args, **kwargs)
#
#         return wrapper
#
#     return decorator
#
#
# def raise_for_status_with_text(response: httpx.Response) -> None:
#     """Raise an error with the response text."""
#     try:
#         response.raise_for_status()
#     except HTTPStatusError as e:
#         raise ValueError(response.text) from e
#
#
# def get_enum_value(enum: Union[Enum, str]) -> str:
#     """Get the value of a string enum."""
#     if isinstance(enum, Enum):
#         return enum.value
#     return enum
#
#
# @lru_cache
# def get_runtime_environment() -> dict:
#     """Get information about the environment."""
#     # Lazy import to avoid circular imports
#     from parea import version
#
#     return {
#         "sdk_version": version,
#         "library": "langsmith",
#         "platform": platform.platform(),
#         "runtime": "python",
#         "runtime_version": platform.python_version(),
#     }
#
# def _is_localhost(url: str) -> bool:
#     """Check if the URL is localhost."""
#     try:
#         netloc = urlsplit(url).netloc.split(":")[0]
#         ip = socket.gethostbyname(netloc)
#         return ip == "127.0.0.1" or ip.startswith("0.0.0.0") or ip.startswith("::")
#     except socket.gaierror:
#         return False
#
#
# ID_TYPE = Union[UUID, str]
#
#
# def _serialize_json(obj: Any) -> str:
#     """Serialize an object to JSON."""
#     if isinstance(obj, datetime):
#         return obj.isoformat()
#     else:
#         return str(obj)
#
#
# async def close_session(session: httpx.AsyncClient) -> None:
#     """Close the session."""
#     logger.debug("Closing Client.session")
#     await session.aclose()
#
#
# def _validate_api_key_if_hosted(api_url: str, api_key: Optional[str]) -> None:
#     """Verify API key is provided if url not localhost."""
#     if not _is_localhost(api_url):
#         if not api_key:
#             raise PareaUserError("API key must be provided when using hosted LangSmith API")
#
#
# def _get_api_key(api_key: Optional[str]) -> Optional[str]:
#     api_key = api_key if api_key is not None else os.getenv("DEV_API_KEY")
#     if api_key is None or not api_key.strip():
#         return None
#     return api_key.strip().strip('"').strip("'")
#
#
# def _get_api_url(api_url: Optional[str], api_key: Optional[str]) -> str:
#     _api_url = api_url if api_url is not None else "http://localhost:8000/api/parea/v1"
#     if not _api_url.strip():
#         raise PareaUserError("Parea API URL cannot be empty")
#     return _api_url.strip().strip('"').strip("'").rstrip("/")
#
#
# class Client:
#     """Client for interacting with the LangSmith API."""
#
#     def __init__(
#         self,
#         api_url: Optional[str] = None,
#         *,
#         api_key: Optional[str] = None,
#     ) -> None:
#         """Initialize a Client instance."""
#         self.api_key = _get_api_key(api_key)
#         self.api_url = _get_api_url(api_url, self.api_key)
#         _validate_api_key_if_hosted(self.api_url, self.api_key)
#
#         # Create a session and register a finalizer to close it
#         self.session = httpx.AsyncClient()
#         weakref.finalize(self, close_session, self.session)
#
#     def _repr_html_(self) -> str:
#         """Return an HTML representation of the instance with a link to the URL."""
#         link = self._host_url
#         return f'<a href="{link}", target="_blank" rel="noopener">LangSmith Client</a>'
#
#     def __repr__(self) -> str:
#         """Return a string representation of the instance with a link to the URL."""
#         return f"Client (API URL: {self.api_url})"
#
#     @property
#     def _host_url(self) -> str:
#         """The web host url."""
#         if _is_localhost(self.api_url):
#             link = "http://localhost"
#         elif "dev" in self.api_url.split(".", maxsplit=1)[0]:
#             link = "http://localhost"
#         else:
#             link = "https://optimus-prompt-backend.vercel.app"
#         return link
#
#     @property
#     def _headers(self) -> Dict[str, str]:
#         """Get the headers for the API request."""
#         headers = {}
#         if self.api_key:
#             headers["x-api-key"] = self.api_key
#         return headers
#
#     async def request_with_retries(
#         self,
#         request_method: str,
#         url: str,
#         request_kwargs: Mapping,
#     ) -> httpx.Response:
#         """Send a request with retries."""
#
#         try:
#             response = await self.session.request(request_method, url, **request_kwargs)
#             raise_for_status_with_text(response)
#             return response
#         except HTTPStatusError as e:
#             if response.status_code == 500:
#                 raise PareaAPIError(f"Server error caused failure to {request_method} {url} in Parea API. {e}")
#             else:
#                 raise PareaUserError(f"Failed to {request_method} {url} in LangSmith API. {e}")
#         except httpx.ConnectError as e:
#             raise PareaConnectionError(f"Connection error caused failure to {request_method} {url} in Parea API. Please confirm your PAREA_ENDPOINT.") from e
#         except Exception as e:
#             raise PareaError(f"Failed to {request_method} {url} in Parea API. {e}") from e
#
#     async def _get_with_retries(self, path: str, params: Optional[Dict[str, Any]] = None) -> httpx.Response:
#         """Send a GET request with retries."""
#         return await self.request_with_retries(
#             "get",
#             f"{self.api_url}/{path}",
#             request_kwargs={
#                 "params": params,
#                 "headers": self._headers,
#             },
#         )
#
#     async def _get_paginated_list(self, path: str, *, params: Optional[dict] = None) -> AsyncIterator[dict]:
#         """Get a paginated list of items."""
#         params_ = params.copy() if params else {}
#         offset = params_.get("offset", 0)
#         params_["limit"] = params_.get("limit", 100)
#         while True:
#             params_["offset"] = offset
#             response = await self._get_with_retries(path, params=params_)
#             items = response.json()
#             if not items:
#                 break
#             for item in items:
#                 yield item
#             if len(items) < params_["limit"]:
#                 break
#             offset += len(items)
#
#     async def create_run(
#         self,
#         name: str,
#         inputs: Dict[str, Any],
#         run_type: str,
#         *,
#         execution_order: Optional[int] = None,
#         **kwargs: Any,
#     ) -> None:
#         """Persist a run to the LangSmith API."""
#
#         project_name = kwargs.pop(
#             "project_name",
#             kwargs.pop(
#                 "session_name",
#                 os.environ.get(
#                     "PAREA_PROJECT",
#                     os.environ.get("PAREA_SESSION", "default"),
#                 ),
#             ),
#         )
#         run_create = {
#             **kwargs,
#             "session_name": project_name,
#             "name": name,
#             "inputs": inputs,
#             "run_type": run_type,
#             "execution_order": execution_order if execution_order is not None else 1,
#         }
#         run_extra = cast(dict, run_create.setdefault("extra", {}))
#         runtime = run_extra.setdefault("runtime", {})
#         runtime_env = get_runtime_environment()
#         run_extra["runtime"] = {**runtime_env, **runtime}
#         headers = {**self._headers, "Accept": "application/json"}
#         await self.request_with_retries(
#             "post",
#             f"{self.api_url}/runs",
#             request_kwargs={
#                 "json": run_create,
#                 "headers": headers,
#             },
#         )
#
#     async def completion(self, data: Completion) -> CompletionResponse:
#         headers = {**self._headers, "Accept": "application/json"}
#         r = await self.request_with_retries(
#             "post",
#             f"{self.api_url}/completion",
#             request_kwargs={
#                 "json": asdict(data),
#                 "headers": headers,
#             },
#         )
#         return CompletionResponse(**r.json())
#
#     async def update_run(
#         self,
#         run_id: ID_TYPE,
#         **kwargs: Any,
#     ) -> None:
#         """Update a run in the LangSmith API."""
#         headers = {**self._headers, "Accept": "application/json"}
#         await self.request_with_retries(
#             "patch",
#             f"{self.api_url}/runs/{run_id}",
#             request_kwargs={
#                 "json": kwargs,
#                 "headers": headers,
#             },
#         )
