import json
from pathlib import Path

import yaml
from dask.distributed import Client, LocalCluster
from pydantic import create_model
from graphviz import Source

from irrigation_processor.pipeline import XcubeDataStoreStorage
from src.irrigation_processor.pipeline import FileStorage, LocalService, Pipeline
from src.irrigation_processor.steps import step


def run_pipeline(config_file: Path,
                 disable_steps: list[str]=None):
    registry = step.get_registry()

    if disable_steps:
        for s in disable_steps:
            registry.disable(s)

    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")

    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    base_cfg = config.get("base", {})

    for step_name, step_cfg in config.items():
        if step_name == "base":
            continue

        step_meta = registry.get(step_name)

        if not step_meta:
            raise ValueError(f"Step not found: {step_name}")

        merged_cfg = {**base_cfg, **step_cfg}

        config_model = create_model(
            f"{step_name.capitalize()}Config",
            **{k: (type(v), v) for k, v in merged_cfg.items()},
        )

        context_obj = config_model(**merged_cfg)
        step_meta.context_cls = lambda obj=context_obj: obj

    storage = XcubeDataStoreStorage()
    service = LocalService(storage=storage, use_cache=True)
    p = Pipeline(service=service, pipeline_name="irrigation_estimates")

    p.add_steps_from_registry(registry)
    dot_str = p.visualize_dot()

    src = Source(dot_str)
    src.render("pipeline", format="png", view=True)
    state = p.run()
    print("State metadata:\n", json.dumps(state, indent=2))


if __name__ == "__main__":
    # cluster = LocalCluster(n_workers=4, threads_per_worker=2)
    # client = Client(cluster)
    disbaled_steps = ["calibration", "simulation"]
    config_path = Path("config.yml")
    run_pipeline(
        config_file=config_path,
        disable_steps=disbaled_steps
    )

# TODO:
#  Meet with Norman
#  Allow to choose which step to run, list steps, etc. [DONE]
#  Add ./pipeline_cache/{step}.json as cache of each step (every run will
#   overwrite?) to allow running any step instead of full pipeline for
#   debugging (if possible) [DONE]
#  depends_on should also factor into building the graph [DONE]
#  Add Typer CLI to list pipelines, steps, execute
#  Add config.yml and use that to create the context [DONE]
#  Add tests
#  Add documentation