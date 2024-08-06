from typing import Any, Dict, List, Optional, Union, cast

import datetime
import uuid

from langchain_core.tracers.schemas import Run

from parea.parea_logger import parea_logger
from parea.schemas.log import TraceIntegrations
from parea.utils.trace_integrations.langchain_utils import RunLikeDict, _as_uuid, _dumps_json, deepish_copy


class PareaLangchainClient:
    created: List[RunLikeDict] = []
    updated: List[RunLikeDict] = []
    _langchain_to_parea_root_data: Dict[str, Dict[str, str]] = {}

    def __init__(
        self,
        session_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        end_user_identifier: Optional[str] = None,
        deployment_id: Optional[str] = None,
    ) -> None:
        self.project_uuid: Optional[str] = parea_logger.get_project_uuid()
        self._session_id = session_id
        self._tags = tags
        self._metadata = metadata
        self._end_user_identifier = end_user_identifier
        self._deployment_id = deployment_id

    def create_run(self, *args, **kwargs) -> None:
        pass

    def create_run_trace(self, run: Run):
        run_create = self._run_transform(run, copy=True)
        self.created.append(run_create)

    @staticmethod
    def _run_transform(
        run: Union[Run, dict, RunLikeDict],
        update: bool = False,
        copy: bool = False,
        for_log: bool = False,
    ) -> dict:
        """Transform the given run object into a dictionary representation.
        Args:
            run (Union[ls_schemas.Run, dict]): The run object to transform.
            update (bool, optional): Whether to update the run. Defaults to False.
            copy (bool, optional): Whether to copy the run. Defaults to False.
        Returns:
            dict: The transformed run object as a dictionary.
        """
        if hasattr(run, "dict") and callable(getattr(run, "dict")):
            run_create: dict = run.dict()
        else:
            run_create = cast(dict, run)
        if "id" not in run_create:
            run_create["id"] = uuid.uuid4()
        elif isinstance(run_create["id"], str):
            run_create["id"] = uuid.UUID(run_create["id"])
        if "inputs" in run_create and run_create["inputs"] is not None:
            if copy:
                run_create["inputs"] = deepish_copy(run_create["inputs"])
            run_create["inputs"] = run_create["inputs"]
        if "outputs" in run_create and run_create["outputs"] is not None:
            if copy:
                run_create["outputs"] = deepish_copy(run_create["outputs"])
            run_create["outputs"] = run_create["outputs"]
        if not update and not run_create.get("start_time"):
            run_create["start_time"] = datetime.datetime.now(datetime.timezone.utc)
        if not run_create.get("end_time", None) and for_log:
            run_create["end_time"] = datetime.datetime.now(datetime.timezone.utc)
        if not run_create.get("children", None):
            run_create["children"] = []
        return run_create

    def update_run(self, *args, **kwargs) -> None:
        pass

    def update_run_trace(self, run: Run) -> None:
        run_update = self._run_transform(run, update=True)
        self.updated.append(run_update)

    def log(self):
        create_dicts = [self._run_transform(run, for_log=True) for run in self.created or []]
        update_dicts = [self._run_transform(run, update=True, for_log=True) for run in self.updated or []]

        root_id = ""
        create_by_id = {run["id"]: run for run in create_dicts}
        for run in create_dicts:
            if not root_id and run["id"] == run["trace_id"]:
                root_id = run["id"]

            parent_run_id = run.get("parent_run_id")
            if parent_run_id and parent_run_id in create_by_id:
                children = create_by_id[parent_run_id]["children"]
                if str(run["id"]) not in children:
                    children.append(str(run["id"]))

        # combine post and patch dicts where possible
        if update_dicts and create_dicts:
            standalone_updates: list[dict] = []
            for run in update_dicts:
                if run["id"] in create_by_id:
                    create_by_id[run["id"]].update({k: v for k, v in run.items() if v is not None})
                else:
                    standalone_updates.append(run)
            update_dicts = standalone_updates

        root_trace = create_by_id.get(root_id, None)
        root_output = root_trace.get("outputs", None) if root_trace else None
        root_children = root_trace.get("children", []) if root_trace else []
        if not root_output and root_children:
            root_output = create_by_id.get(_as_uuid(root_children[-1]), {}).get("outputs", None)

        for run in create_dicts:
            if root_id == run["id"]:
                run["outputs"] = root_output

            self.process_log(run)

        if update_dicts:
            for run in update_dicts:
                self.process_log(run)

    def stream_log(self, run: Run) -> None:
        streaming_run: dict = self._run_transform(run, for_log=True)
        self.process_log(streaming_run)

    def process_log(self, run: dict) -> None:
        run = self._fill_run_with_parea_trace_data(run)
        parea_logger.record_vendor_log(_dumps_json(run), TraceIntegrations.LANGCHAIN)

    def set_parea_root_and_parent_trace_id(self, langchain_to_parea_root_data: dict) -> None:
        self._langchain_to_parea_root_data.update(langchain_to_parea_root_data)

    def _fill_run_with_parea_trace_data(self, run: dict) -> dict:
        is_root = (run.get("execution_order", None) and run["execution_order"] == 1) or not run.get("parent_run_id", None)
        data = self._langchain_to_parea_root_data.get(run["id"] if is_root else run["trace_id"], {})
        if is_root:
            run["parent_run_id"] = data.get("parent_trace_id", run["parent_run_id"])
            run["_session_id"] = self._session_id
            run["_tags"] = self._tags
            run["_metadata"] = self._metadata
            run["_end_user_identifier"] = self._end_user_identifier
            run["_deployment_id"] = self._deployment_id

        run["trace_id"] = data.get("root_trace_id", run["trace_id"])
        run["project_uuid"] = self.project_uuid
        run["experiment_uuid"] = data.get("experiment_uuid")
        return run
