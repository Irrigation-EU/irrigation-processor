from irrigation_processor.constants import LOG

from .service import LocalService
from .step import FromStep, StepMeta, StepRegistry


class Pipeline:
    def __init__(self, service: LocalService, pipeline_name: str):
        self.steps: dict[str, StepMeta] = {}
        self.service = service
        self.pipeline_name = pipeline_name

    def add(self, step_meta: StepMeta):
        if step_meta.name in self.steps:
            raise KeyError(f"step {step_meta.name} already added")
        self.steps[step_meta.name] = step_meta

    def add_steps_from_registry(self, registry: StepRegistry):
        for meta in registry.all():
            self.add(meta)

    def _build_graph(self) -> dict[str, set[str]]:
        deps: dict[str, set[str]] = {name: set() for name in self.steps}
        for name, meta in self.steps.items():
            deps[name].update(meta.depends_on)
            for inp in meta.inputs:
                if isinstance(inp, FromStep):
                    deps[name].add(inp.step)

        for step, srcs in deps.items():
            for dep in srcs:
                if dep not in self.steps:
                    raise ValueError(f"Step '{step}' depends on unknown step '{dep}'")

        return deps

    @staticmethod
    def _toposort(deps: dict[str, set[str]]) -> list[str]:
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
        if not self.steps:
            LOG.error("Please add steps to the pipeline first.")
            return None
        deps = self._build_graph()
        order = Pipeline._toposort(deps)
        return self.service.run(self.pipeline_name, order, self.steps)
