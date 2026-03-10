import inspect
import json
import os
from typing import Any

from dask.distributed import Client, LocalCluster

from irrigation_processor.config import AppConfig
from irrigation_processor.constants import LOG, PIPELINE_RESULTS_DIR
from irrigation_processor.core.step import FromStep, StepMeta
from irrigation_processor.core.storage import Storage


class LocalService:
    """
    Runs all steps of a pipeline on your local machine.

    It executes each step in order, passes outputs between steps,
    and automatically saves results using the configured storage.
    """

    def __init__(self, storage: Storage, app_config: AppConfig):
        self.storage = storage
        self.app_config = app_config
        # state mapping step_name -> output_key -> metadata dict
        # Metadata dict format:
        # {"type": "inline", "value": ...}
        # {"type": "stored", "data_id": ...}
        self._state: dict[str, dict[str, dict[str, Any]]] = {}

        self.client = None
        self.cluster = None

    def run(
        self,
        pipeline_name: str,
        order: list[str],
        steps: dict[str, StepMeta],
    ):
        """
        Execute steps in given order, stores the intermediate results via
        storage if applicable, and returns a dictionary containing the stored
        outputs for each step.

        Args:
            pipeline_name: Name of the pipeline being executed.
            order: List of step names in execution order.
            steps: Mapping of step names to their metadata definitions.

        Returns:
            A dictionary containing the stored outputs for each step,
            structured as:
                {step_name: {output_key: metadata_dict}}
        """
        LOG.info(f"Starting pipeline: {pipeline_name} from LocalService")

        try:
            for step_name in order:
                LOG.info(f"Running step: {step_name}")
                step_meta = steps[step_name]

                LOG.warning("Restarting Dask client")
                dask_kwargs = self.app_config.dask.dask_kwargs.model_dump()
                self.cluster = LocalCluster(**dask_kwargs)
                self.client = Client(self.cluster)
                LOG.info(f"Initialized Dask cluster: {self.client.dashboard_link}")

                resolved_args, resolved_kwargs = self._resolve_inputs(
                    step_name, step_meta
                )

                sig = inspect.signature(step_meta.func)
                ctx = self.app_config

                call_args: list[Any] = [ctx]

                if "storage" in sig.parameters:
                    call_args.append(self.storage)

                if "dask_client" in sig.parameters:
                    call_args.append(self.client)

                call_args.extend(resolved_args)

                result = step_meta.func(*call_args, **resolved_kwargs)

                out_map = self._normalize_outputs(step_name, step_meta, result)
                self._state[step_name] = out_map
                LOG.info(f"Step state: {step_name}: {out_map}")
                save_pipeline_step_state(pipeline_name, step_name, out_map)

        finally:
            if self.client:
                self.client.close()
            if self.cluster:
                self.cluster.close()

        LOG.info(f"Pipeline run for: {pipeline_name} completed.")
        return self._state

    def _resolve_inputs(self, step_name, meta):
        resolved_args, resolved_kwargs = [], {}
        if isinstance(meta.inputs, (list, tuple)):
            for inp in meta.inputs:
                if isinstance(inp, FromStep):
                    val = self._load_from_state(inp.step, inp.key, step_name)
                    resolved_args.append(val)
                else:
                    resolved_args.append(inp)
        elif isinstance(meta.inputs, dict):
            for name, inp in meta.inputs.items():
                if isinstance(inp, FromStep):
                    val = self._load_from_state(inp.step, inp.key, step_name)
                    resolved_kwargs[name] = val
                else:
                    resolved_kwargs[name] = inp
        return resolved_args, resolved_kwargs

    def _load_from_state(self, step: str, key: str, current_step: str) -> Any:
        if step not in self._state or key not in self._state[step]:
            raise KeyError(
                f"Missing output '{key}' from step '{step}' required by '{current_step}'"
            )

        meta = self._state[step][key]
        return self._load_value(meta)

    def _normalize_outputs(self, step_name: str, meta: StepMeta, result: Any) -> dict:
        out_map = {}

        if isinstance(result, dict):
            if meta.outputs:
                if list(result.keys()) != list(meta.outputs):
                    raise ValueError(
                        f"Output keys/order mismatch for step '{step_name}'. "
                        f"Expected {list(meta.outputs)}, got {list(result.keys())}"
                    )
                out_map.update(result)
            else:
                out_map["return_value"] = result

        elif isinstance(result, (list, tuple)):
            if meta.outputs:
                if len(result) != len(meta.outputs):
                    raise ValueError(
                        f"Step '{step_name}' returned {len(result)} items, "
                        f"but {len(meta.outputs)} outputs were expected."
                    )
                out_map.update(dict(zip(meta.outputs, result)))
            else:
                out_map["return_value"] = result

        else:
            if meta.outputs:
                if len(meta.outputs) != 1:
                    raise ValueError(
                        f"Step '{step_name}' returned a single value, "
                        f"but {len(meta.outputs)} outputs were expected."
                    )
                out_map[meta.outputs[0]] = result
            else:
                out_map["return_value"] = result

        # Store data
        stored_map = {}
        for key, val in out_map.items():
            try:
                stored_map[key] = self._store_value(val, key)
            except Exception as e:
                raise RuntimeError(
                    f"Failed to store output '{key}' of step '{step_name}': {e}"
                ) from e

        return stored_map

    def _store_value(self, val: Any, key: str) -> dict:
        """Recursively store a value and return its metadata identification."""
        if self._is_inline(val):
            return {"type": "inline", "value": val}

        if isinstance(val, (list, tuple)):
            list_items = [self._store_value(v, f"{key}.{i}") for i,
            v in enumerate(val)]
            return {
                "type": "list" if isinstance(val, list) else "tuple",
                "items": list_items,
            }

        if isinstance(val, dict):
            dict_items = {str(k): self._store_value(v, f"{key}.{k}") for k,
            v in val.items()}
            return {"type": "dict", "items": dict_items}

        # For xarray datasets or other heavy objects, we use storage
        # Append .zarr as it's the default format for xcube data store
        data_id = f"{key}.zarr"
        self.storage.save(data_id, val)
        return {"type": "stored", "data_id": data_id}

    def _load_value(self, meta: dict) -> Any:
        """Recursively load a value from its metadata identification."""
        m_type = meta.get("type")
        if m_type == "inline":
            return meta["value"]
        if m_type == "stored":
            return self.storage.load(meta["data_id"])
        if m_type in ("list", "tuple"):
            items = [self._load_value(item) for item in meta["items"]]
            return list(items) if m_type == "list" else tuple(items)
        if m_type == "dict":
            return {k: self._load_value(v) for k, v in meta["items"].items()}
        raise ValueError(f"Unknown metadata type: {m_type}")

    def _is_inline(self, obj: Any) -> bool:
        """
        Recursively check if an object and its contents are simple enough
        to be stored inline in the state file.
        """
        if isinstance(obj, (int, float, str, bool, type(None))):
            return True
        if isinstance(obj, (list, tuple)):
            return all(self._is_inline(item) for item in obj)
        if isinstance(obj, dict):
            return all(
                self._is_inline(k) and self._is_inline(v) for k, v in obj.items()
            )
        return False


def save_pipeline_step_state(pipeline_name: str, step_name: str, data: dict) -> str:
    base_path = os.path.join(PIPELINE_RESULTS_DIR, pipeline_name)
    os.makedirs(base_path, exist_ok=True)
    file_path = os.path.join(base_path, f"{step_name}.json")

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)

    return file_path
