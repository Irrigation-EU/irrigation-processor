from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence


@dataclass
class FromStep:
    """
    A helper class used to reference an output value produced by another step.

    Use this in a step's inputs to say:
    "Take the value `key` from step `step`."
    """

    step: str
    key: str

    def to_dict(self) -> dict:
        return {"step": self.step, "key": self.key}


InputValue = Any | FromStep
Inputs = Sequence[InputValue] | Mapping[str, InputValue]


class StepRegistry:
    """
    Stores and manages all available pipeline steps.

    Steps are registered using the @registry.step decorator.
    """

    def __init__(self):
        self._steps = {}

    def register(self, step_meta: "StepMeta"):
        if step_meta.name in self._steps:
            raise KeyError(f"A step named '{step_meta.name}' is already registered")
        self._steps[step_meta.name] = step_meta

    def all(self):
        return list(self._steps.values())

    def get(self, step_name: str) -> "StepMeta":
        if step_name not in self._steps:
            raise KeyError(f"No step named '{step_name}' is registered")
        return self._steps[step_name]

    def step(
        self,
        func: Callable | None = None,
        /,
        *,
        inputs: Inputs = (),
        outputs: Sequence[str] = (),
        depends_on: Sequence[str] = (),
        name: str | None = None,
    ) -> Callable:
        """
        Decorator used to define and register a pipeline step.

        Args:
            func: The function to decorate. Usually omitted when using
                the decorator with arguments.
            inputs: Input values for the step. Can be literals or
                references to other steps via `FromStep`.
            outputs: Optional names of the outputs produced by the step.
            depends_on: Optional names of steps that must run before this step.
            name: Optional custom name for the step. Defaults to
                the function name.

        Returns:
            The original function, registered as a pipeline step.

        Example:
            @registry.step(
                inputs=[FromStep("load_data", "dataset")],
                outputs=["result"],
                depends_on=["load_data"],
            )
            def process(ctx, dataset):
                ...
        """

        def decorator(f: Callable) -> Callable:
            meta = StepMeta(
                func=f,
                inputs=inputs,
                outputs=outputs,
                depends_on=depends_on,
                name=name,
            )
            self.register(meta)
            return f

        if func is None:
            return decorator
        return decorator(func)


class StepMeta:
    """
    Describes a pipeline step: its function, inputs, outputs,
    and dependencies.
    """

    def __init__(
        self,
        func: Callable,
        inputs: Inputs = (),
        outputs: Sequence[str] = (),
        depends_on: Sequence[str] = (),
        name: str | None = None,
    ):
        self.func = func
        self.func_path = f"{func.__module__}:{func.__name__}"
        self.name = name or func.__name__
        self.inputs = inputs
        self.outputs = outputs
        self.depends_on = tuple(depends_on)

    def __repr__(self):
        return (
            f"StepMeta(name={self.name}, "
            f"func_path={self.func_path}, inputs={self.inputs}, "
            f"outputs={self.outputs}, depends_on={self.depends_on}), "
        )
