from dataclasses import dataclass
from typing import Callable, Iterable, Sequence, Any, Mapping


@dataclass
class FromStep:
    step: str
    key: str

    def to_dict(self) -> dict:
        return {"step": self.step, "key": self.key}

InputValue = Any | FromStep
Inputs = Sequence[InputValue] | Mapping[str, InputValue]

class StepRegistry:
    def __init__(self):
        self._steps = {}
        self._disabled_steps = set()

    def register(self, step_meta: "StepMeta"):
        if step_meta.name in self._steps:
            raise KeyError(f"A step named '{step_meta.name}' is already registered")
        self._steps[step_meta.name] = step_meta

    def all(self, include_disabled: bool = False):
        if include_disabled:
            return list(self._steps.values())
        return [
            step
            for name, step in self._steps.items()
            if name not in self._disabled_steps
        ]

    def get(self, step_name: str) -> "StepMeta":
        if step_name not in self._steps:
            raise KeyError(f"No step named '{step_name}' is registered")
        return self._steps[step_name]

    def disable(self, step_name: str):
        if step_name not in self._steps:
            raise KeyError(f"No step named '{step_name}' is registered")
        self._disabled_steps.add(step_name)

    def enable(self, step_name: str):
        if step_name not in self._steps:
            raise KeyError(f"No step named '{step_name}' is registered")
        self._disabled_steps.discard(step_name)

    def step(
        self,
        func: Callable | None = None,
        /,
        *,
        inputs: Inputs = (),
        outputs: Sequence[str] = (),
        depends_on: Sequence[str] = (),
        name: str | None = None,
        context_cls: type | None = None,
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
            self.register(meta)
            return f

        if func is None:
            return decorator
        return decorator(func)

class StepMeta:
    def __init__(
        self,
        func: Callable,
        inputs: Inputs = (),
        outputs: Sequence[str] = (),
        depends_on: Sequence[str] = (),
        name: str | None = None,
        context_cls: type | None = None,
    ):
        self.func = func
        self.func_path = f"{func.__module__}:{func.__name__}"
        self.name = name or func.__name__
        self.inputs = inputs
        self.outputs = outputs
        self.depends_on = tuple(depends_on)
        self.context_cls = context_cls

    def __repr__(self):
        return (
            f"StepMeta(name={self.name}, "
            f"func_path={self.func_path}, inputs={self.inputs}, "
            f"outputs={self.outputs}, depends_on={self.depends_on}), "
        )
