from typing import Any, Dict, Optional

import json
import logging
import os

from attrs import asdict, define, field
from cattrs import structure

from parea.api_client import HTTPClient
from parea.constants import PAREA_OS_ENV_EXPERIMENT_UUID
from parea.helpers import serialize_metadata_values
from parea.schemas.log import TraceIntegrations
from parea.schemas.models import CreateGetProjectResponseSchema, TraceLog, UpdateLog
from parea.utils.universal_encoder import json_dumps

logger = logging.getLogger()

LOG_ENDPOINT = "/trace_log"
VENDOR_LOG_ENDPOINT = "/trace_log/{vendor}"


@define
class PareaLogger:
    _client: HTTPClient = field(init=False, default=None)
    _project_uuid: str = field(init=False, default=None)
    _project_name: str = field(init=False, default=None)

    def set_client(self, client: HTTPClient) -> None:
        self._client = client

    def set_project_uuid(self, project_uuid: str, project_name: str) -> None:
        self._project_uuid = project_uuid
        self._project_name = project_name

    def _get_project_uuid(self) -> Optional[str]:
        if not self._project_uuid:
            self._project_uuid = self._create_or_get_project(self._project_name or "default").uuid
        try:
            return self._project_uuid
        except Exception as e:
            logger.error(f"PareaLogger: Error getting project uuid for project {self._project_name}: {e}")
            return None

    def _create_or_get_project(self, name: str) -> CreateGetProjectResponseSchema:
        r = self._client.request(
            "POST",
            "/project",
            data={"name": name},
        )
        return structure(r.json(), CreateGetProjectResponseSchema)

    def update_log(self, data: UpdateLog) -> None:
        data = serialize_metadata_values(data)
        self._client.request(
            "PUT",
            LOG_ENDPOINT,
            data=asdict(data),
        )

    def record_log(self, data: TraceLog) -> None:
        data = serialize_metadata_values(data)
        data.project_uuid = self._get_project_uuid()
        self._client.request(
            "POST",
            LOG_ENDPOINT,
            data=asdict(data),
        )

    async def arecord_log(self, data: TraceLog) -> None:
        data = serialize_metadata_values(data)
        data.project_uuid = self._project_uuid
        await self._client.request_async(
            "POST",
            LOG_ENDPOINT,
            data=asdict(data),
        )

    def default_log(self, data: TraceLog) -> None:
        if self._client:
            if data.target:
                data.target = json_dumps(data.target)
            self.record_log(data)

    def record_vendor_log(self, data: Dict[str, Any], vendor: TraceIntegrations) -> None:
        data["project_uuid"] = self._get_project_uuid()
        if experiment_uuid := os.getenv(PAREA_OS_ENV_EXPERIMENT_UUID, None):
            data["experiment_uuid"] = experiment_uuid
        self._client.add_integration("langchain")
        self._client.request(
            "POST",
            VENDOR_LOG_ENDPOINT.format(vendor=vendor.value),
            data=json.loads(json_dumps(data)),  # uuid is not serializable
        )

    async def arecord_vendor_log(self, data: Dict[str, Any], vendor: TraceIntegrations) -> None:
        data["project_uuid"] = self._get_project_uuid()
        if experiment_uuid := os.getenv(PAREA_OS_ENV_EXPERIMENT_UUID, None):
            data["experiment_uuid"] = experiment_uuid
        self._client.add_integration("langchain")
        await self._client.request_async(
            "POST",
            VENDOR_LOG_ENDPOINT.format(vendor=vendor.value),
            data=json.loads(json_dumps(data)),  # uuid is not serializable
        )


parea_logger = PareaLogger()
