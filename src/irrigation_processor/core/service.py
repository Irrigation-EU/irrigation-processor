import inspect
import json
import os
from typing import Any

from dask.distributed import Client, LocalCluster

from irrigation_processor.constants import LOG, PIPELINE_RESULTS_CACHE_DIR
from irrigation_processor.core.step import StepMeta, FromStep
from irrigation_processor.core.storage import Storage


class Service:
    def __init__(self, storage: Storage, use_cache: bool = False):
        self.storage = storage
        # state mapping step_name -> output_key -> metadata (returned by storage.save)
        self._state: dict[str, dict[str, dict[str, Any]]] = {}
        self.use_cache = use_cache

    def run(self,
            pipeline_name: str,
            order: list[str],
            steps: dict[str, StepMeta]
            ):
        """Execute steps in given order. Returns state with outputs."""
        raise NotImplementedError


class LocalService(Service):
    def run(self,
            pipeline_name: str,
            order: list[str],
            steps: dict[str, StepMeta],
            ):
        LOG.info(f"Starting pipeline: {pipeline_name} from LocalService")
        for step_name in order:
            step_meta = steps[step_name]
            LOG.info(f"Running step: {step_name}")

            resolved_args, resolved_kwargs = self._resolve_inputs(
                step_name, step_meta, self.storage, pipeline_name
            )

            sig = inspect.signature(step_meta.func)
            if "dask_client" in sig.parameters:
                cluster = LocalCluster(
                    n_workers=4,
                    threads_per_worker=1,
                    memory_limit="4GB",
                )
                client = Client(cluster)
                resolved_kwargs["dask_client"] =  client

            ctx = (
                step_meta.context_cls()
                if step_meta.context_cls
                else None
            )

            result = step_meta.func(ctx, *resolved_args, **resolved_kwargs)
            out_map = self._normalize_outputs(step_name, step_meta, result)
            self._state[step_name] = out_map
            LOG.info(f"Step state: {step_name}: {out_map}")
            save_pipeline_step_state(pipeline_name, step_name, out_map)
            if "dask_client" in sig.parameters:
                client.close()

        LOG.info(f"Pipeline run for: {pipeline_name} completed.")
        return self._state

    def _resolve_inputs(self, step_name, meta, storage, pipeline_name):
        resolved_args, resolved_kwargs = [], {}
        if isinstance(meta.inputs, (list, tuple)):
            for inp in meta.inputs:
                if isinstance(inp, FromStep):
                    s, k = inp.step, inp.key

                    # check if previous steps results are available in cache
                    if self.use_cache:
                        LOG.info(f"Using cache for args for step: {s}")
                        state = load_pipeline_step_state(pipeline_name,
                                                        s)
                        resolved_args.append(storage.load(state[k]))
                    # if not using cache, checking if previous steps ran and
                    # expected output exists
                    else:
                        if s not in self._state or k not in self._state[s]:
                            raise KeyError(
                                f"Missing output '{k}' from step '{s}' required by '{step_name}'"
                            )
                        resolved_args.append(storage.load(self._state[s][k]))
                else:
                    resolved_args.append(inp)
        elif isinstance(meta.inputs, dict):
            for name, inp in meta.inputs.items():
                if isinstance(inp, FromStep):
                    s, k = inp.step, inp.key

                    # check if previous steps results are available in cache
                    if self.use_cache:
                        LOG.info(f"Using cache for kwargs for step:"
                                    f" {s}")
                        state = load_pipeline_step_state(pipeline_name, s)
                        resolved_kwargs[name] = storage.load(state[k])

                    # if not using cache, checking if previous steps ran and
                    # expected output exists
                    else:
                        if s not in self._state or k not in self._state[s]:
                            raise KeyError(
                                f"Missing output '{k}' from step '{s}' required by '{step_name}'"
                            )
                        resolved_kwargs[name] = storage.load(self._state[s][k])
                else:
                    resolved_kwargs[name] = inp
        return resolved_args, resolved_kwargs

    def _normalize_outputs(
            self,
            step_name: str,
            meta: StepMeta,
            result: Any
    ) -> dict:
        out_map = {}

        if isinstance(result, dict):
            if meta.outputs:
                result_keys = list(result.keys())
                expected_keys = list(meta.outputs)

                if result_keys != expected_keys:
                    raise ValueError(
                        f"Output keys/order mismatch for step '{step_name}'. "
                        f"Expected {expected_keys}, got {result_keys}"
                    )
            out_map.update(result)
        else:
            if meta.outputs:
                LOG.debug(f"{step_name} | {result} | {type(result)}")
                if isinstance(result, (list, tuple)):
                    if len(result) != len(meta.outputs):
                        raise ValueError("The length of the expected outputs: "
                                         f"{len(meta.outputs)} is not the "
                                         f"same as the length: {len(result)} of "
                                         f"retuned iterable by step {step_name}")
                    for i, k in enumerate(meta.outputs):
                        out_map[k] = result[i]
                else:
                    if len(meta.outputs) > 1:
                        raise ValueError("More outputs specified than the step: "
                                    f"{step_name} returned:"
                                    f" {len(meta.outputs)}")
                    out_map[meta.outputs[0]] = result
            else:
                raise ValueError(f"The step {step_name} does not return a "
                                 f"dict nor the output was defined in the "
                                 f"decorator.")

        # Then store the data if any big data found in this json and replace
        # it with its path instead
        stored_map = {}
        for key, val in out_map.items():
            if key == "result":
                key = f"{step_name}_{key}"
            stored_map[key] = self.storage.save(key, val)
        return stored_map



def save_pipeline_step_state(pipeline_name: str, step_name: str, data: dict) -> str:
    base_path = os.path.join(PIPELINE_RESULTS_CACHE_DIR, pipeline_name)
    os.makedirs(base_path, exist_ok=True)
    file_path = os.path.join(base_path, f"{step_name}.json")

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)

    return file_path

def load_pipeline_step_state(pipeline_name: str, step_name: str) -> dict:
    file_path = os.path.join(PIPELINE_RESULTS_CACHE_DIR, pipeline_name,
                             f"{step_name}.json")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"No saved step found at {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)