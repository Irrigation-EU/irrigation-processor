import abc
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Union

import xarray as xr

from irrigation_processor.constants import logger, INPUT_DIR


class StepRegistry:
    def __init__(self):
        self._steps = {}

    def register(self, step_meta: "StepMeta"):
        if step_meta.name in self._steps:
            raise KeyError(f"A step named '{step_meta.name}' is already registered")
        self._steps[step_meta.name] = step_meta

    def all(self):
        return list(self._steps.values())


#
# STEP_REGISTRY = StepRegistry()


@dataclass
class FromTask:
    def __init__(self, step: str, key: str):
        self.step = step
        self.key = key

    def to_dict(self) -> dict:
        return {"step": self.step, "key": self.key}


class StepMeta:
    def __init__(
        self,
        func: Callable,
        inputs: Iterable[str] = (),
        outputs: Iterable[str] = (),
        depends_on: Iterable[str] = (),
        name: Optional[str] = None,
        context_cls: Optional[type] = None,
    ):
        self.func = func
        self.func_path = f"{func.__module__}:{func.__name__}"
        self.name = name or func.__name__
        self.inputs = tuple(inputs)
        self.outputs = tuple(outputs)
        self.depends_on = tuple(depends_on)
        self.context_cls = context_cls

    def __repr__(self):
        return (
            f"StepMeta(name={self.name}, func={self.func}, "
            f"func_path={self.func_path}, inputs={self.inputs}, "
            f"outputs={self.outputs}, depends_on={self.depends_on})"
        )


class Step:
    def __init__(
        self,
    ):
        self.step_registry = StepRegistry()

    def get_registry(self):
        return self.step_registry

    def register(
        self,
        func: Optional[Callable] = None,
        /,
        *,
        inputs: Iterable = (),
        outputs: Iterable[str] = (),
        depends_on: Iterable[str] = (),
        name: Optional[str] = None,
        context_cls: Optional[type] = None,
    ) -> Callable:
        def decorator(f: Callable) -> Callable:
            meta = StepMeta(
                func=f,
                inputs=inputs,
                outputs=outputs,
                depends_on=depends_on,
                name=name,
                context_cls=context_cls,
            )
            self.step_registry.register(meta)
            return f

        if func is None:
            return decorator
        return decorator(func)


class Storage(abc.ABC):
    @abc.abstractmethod
    def save(self, key: str, obj: Any) -> Dict[str, Any]:
        """Save object and return metadata (e.g. where it was saved).

        This method must handle 2 cases:
        1. Python literals - int, float, bool, dict, set, list
        2. big data - xarray datasets, pandas dataframes etc.

        The format to return must be a dict per key

        inline = True means it is a Python literal
        {
                "inline": True,
                "type": type(obj).__name__
                "value": obj,
        }

        inline = False means it is big data

        {
                "inline": False,
                "type": type(obj).__name__,
                "path": "path-to-the-stored-big-data",
                "format": "<your-format-name>",
        }

        This dict is not required to be followed strictly but just as a general
        hint to use as this dict will be completely and only used in your
        implementation of the Storage subclass.

        """

    @abc.abstractmethod
    def load(self, metadata: Dict[str, Any]) -> Any:
        """Load an object previously saved using the metadata returned by save.

        This method must handle the loading of the 3 cases as discussed in
        save() method.
        """


class FileStorage(Storage):
    """Simple filesystem storage.

    - For xarray.Dataset -> saves to zarr
    - Fallback: pickle
    - Small literals (int/float/str/dict) are kept inline in metadata (no file written)
    """

    def __init__(self, root: Union[str, Path]):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _filename_for_key(self, key: str, suffix: str) -> Path:
        safe = key.replace("/", "_")
        return self.root / f"{safe}{suffix}"

    def save(self, key: str, obj: Any) -> Dict[str, Any]:
        if isinstance(obj, (int, float, str, bool)):
            return {"inline": True, "value": obj, "type": type(obj).__name__}

        if isinstance(obj, xr.Dataset):
            fn = self._filename_for_key(key, ".zarr")
            obj.to_zarr(fn)
            return {
                "inline": False,
                "path": str(fn),
                "format": "zarr",
                "type": type(obj).__name__,
            }

        # fallback: pickle everything else
        fn = self._filename_for_key(key, ".pkl")
        with open(fn, "wb") as f:
            pickle.dump(obj, f)
        return {
            "inline": False,
            "path": str(fn),
            "format": "pickle",
            "type": type(obj).__name__,
        }

    def load(self, metadata: Dict[str, Any]) -> Any:
        if metadata.get("inline"):
            return metadata["value"]
        path = metadata.get("path")
        fmt = metadata.get("format")
        if fmt == "zarr":
            return xr.open_dataset(path)
        if fmt == "pickle":
            with open(path, "rb") as f:
                return pickle.load(f)
        raise RuntimeError(f"Unknown storage format: {fmt}")


class XcubeDataStoreStorage(Storage):
    def __init__(self, store_id: str = "file", store_kwargs: dict = {}):
        from xcube.core.store import new_data_store
        if store_id is None and "root" not in store_kwargs:
            store_kwargs.update({"root": INPUT_DIR})
        self.store = new_data_store(store_id, **store_kwargs)


    def save(self, key: str, obj: Any) -> Dict[str, Any]:
        if isinstance(obj, (int, float, str, bool)):
            return {"inline": True, "value": obj, "type": type(obj).__name__}

        if isinstance(obj, xr.Dataset):
            data_id = key + ".zarr"
            self.store.write_data(obj, data_id)
            return {
                "inline": False,
                "data_id": data_id,
                "type": type(obj).__name__
            }

        raise RuntimeError(f"Unknown storage format: {type(obj)}")

    def load(self, metadata: Dict[str, Any]) -> Any:
        if metadata.get("inline"):
            return metadata["value"]

        data_id = metadata.get("data_id")
        if not data_id:
            raise RuntimeError(f"Invalid data_id: {data_id}")

        return self.store.open_data(data_id)


class Service:
    def __init__(self, storage: Storage):
        self.storage = storage
        # state mapping step_name -> output_key -> metadata (returned by storage.save)
        self._state: Dict[str, Dict[str, Dict[str, Any]]] = {}

    def run(self, order, steps):
        """Execute steps in given order. Returns state with outputs."""
        raise NotImplementedError


class LocalService(Service):
    def run(self, order, steps):
        logger.info(":::::::LocalService:::::::")
        for step_name in order:
            step_meta = steps[step_name]
            logger.info(f"Running step: {step_name}")

            resolved_args, resolved_kwargs = self._resolve_inputs(
                step_name,
                step_meta,
                self.storage,
            )

            ctx = (
                step_meta.context_cls(step_name=step_name)
                if step_meta.context_cls
                else None
            )

            result = step_meta.func(ctx, *resolved_args, **resolved_kwargs)

            out_map = self._normalize_outputs(step_name, step_meta, result)
            self._state[step_name] = out_map

        logger.info("Pipeline run completed.")
        return self._state

    def _resolve_inputs(self, step_name, meta, storage):
        resolved_args, resolved_kwargs = [], {}
        if isinstance(meta.inputs, (list, tuple)):
            for inp in meta.inputs:
                if isinstance(inp, FromTask):
                    s, k = inp.step, inp.key
                    if s not in self._state or k not in self._state[s]:
                        raise KeyError(
                            f"Missing output '{k}' from step '{s}' required by '{step_name}'"
                        )
                    resolved_args.append(storage.load(self._state[s][k]))
                else:
                    resolved_args.append(inp)
        elif isinstance(meta.inputs, dict):
            for name, inp in meta.inputs.items():
                if isinstance(inp, FromTask):
                    s, k = inp.step, inp.key
                    if s not in self._state or k not in self._state[s]:
                        raise KeyError(
                            f"Missing output '{k}' from step '{s}' required by '{step_name}'"
                        )
                    resolved_kwargs[name] = storage.load(self._state[s][k])
                else:
                    resolved_kwargs[name] = inp
        return resolved_args, resolved_kwargs

    def _normalize_outputs(self, step_name, meta, result):
        out_map = {}
        if isinstance(result, dict):
            out_map.update(result)
        elif meta.outputs and len(meta.outputs) == 1:
            out_map[meta.outputs[0]] = result
        elif isinstance(result, (list, tuple)) and len(result) == len(meta.outputs):
            for k, v in zip(meta.outputs, result):
                out_map[k] = v
        else:
            if meta.outputs:
                for k in meta.outputs:
                    out_map[k] = result
            else:
                out_map["result"] = result

        stored_map = {}
        for key, val in out_map.items():
            stored_map[key] = self.storage.save(f"{step_name}/{key}", val)
        return stored_map


# class AirflowService(LocalService):
#     def run(self, order, steps):
#         """
#         dag_id = gen_dags(order, steps)
#         result = trigger_dag(dag_id)
#         return result
#         """


class Pipeline:
    def __init__(self, service: Service):
        self.steps: Dict[str, StepMeta] = {}
        self.service = service

    def add(self, step_meta: StepMeta):
        if step_meta.name in self.steps:
            raise KeyError(f"step {step_meta.name} already added")
        self.steps[step_meta.name] = step_meta

    def add_steps_from_registry(self, registry: StepRegistry):
        registry = registry
        for meta in registry.all():
            self.add(meta)

    def _build_graph(self) -> Dict[str, List[str]]:
        deps: Dict[str, List[str]] = {name: [] for name in self.steps}
        for name, meta in self.steps.items():
            for d in meta.depends_on:
                deps[name].append(d)
            for inp in meta.inputs:
                if isinstance(inp, FromTask):
                    deps[name].append(inp.step)
        return deps

    @staticmethod
    def _toposort(deps: Dict[str, List[str]]) -> List[str]:
        # Kahn's algorithm
        incoming = {n: set(srcs) for n, srcs in deps.items()}
        out = []

        ready = [n for n, s in incoming.items() if not s]
        while ready:
            node = ready.pop(0)
            out.append(node)

            for m, srcs in incoming.items():
                if node in srcs:
                    srcs.remove(node)
                    if not srcs:
                        ready.append(m)
        if len(out) != len(incoming):
            missing = set(incoming) - set(out)
            raise RuntimeError(
                f"Cycle detected or missing dependencies; remaining: {missing}"
            )
        return out

    def visualize_dot(self) -> str:
        deps = self._build_graph()
        lines = ["digraph pipeline {", "rankdir=LR;"]
        for node in deps:
            lines.append(f'"{node}";')
        for node, srcs in deps.items():
            for s in srcs:
                lines.append(f'"{s}" -> "{node}";')
        lines.append("}")
        return "\n".join(lines)

    def run(self):
        deps = self._build_graph()
        order = Pipeline._toposort(deps)
        return self.service.run(order, self.steps)
