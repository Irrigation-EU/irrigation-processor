import json
from pathlib import Path

import yaml
from dotenv import load_dotenv

from irrigation_processor.constants import LOG
from irrigation_processor.core import LocalService, Pipeline, XcubeDataStoreStorage
from irrigation_processor.steps import registry
from irrigation_processor.utils import inject_dynamic_context_from_config

load_dotenv()

def execute_pipeline(config_file: Path, disable_steps: list[str] = None):
    if disable_steps:
        for s in disable_steps:
            registry.disable(s)

    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")

    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    storage_config = config.get("storage", {})
    storage = XcubeDataStoreStorage(**storage_config)

    inject_dynamic_context_from_config(config, registry, storage)

    service = LocalService(storage=storage)
    p = Pipeline(service=service, pipeline_name="irrigation_estimates")

    p.add_steps_from_registry(registry)

    # render dag
    # dot_str = p.visualize_dot()
    # src = graphviz.Source(dot_str)
    # src.render("pipeline", format="png", view=True)

    state = p.run()
    LOG.info(f"State metadata:\n{json.dumps(state, indent=2)}")


if __name__ == "__main__":
    config_path = Path("config.yml")
    execute_pipeline(
        config_file=config_path,
    )
