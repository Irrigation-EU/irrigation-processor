import json
import os
from pathlib import Path

import typer
import yaml
from dotenv import load_dotenv
from typing_extensions import Annotated

load_dotenv()

APP_NAME = "irrigation-processor"

app = typer.Typer(help="Irrigation Water Use (IWU) estimates processor CLI.")


@app.command()
def run(
    pipeline_name: str = typer.Argument(
        "irrigation_estimates", help="Name of the pipeline."
    ),
    config: Annotated[
        Path,
        typer.Option(
            "--config",
            "-c",
            help="Path to the configuration YAML file.",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
        ),
    ] = Path("config.yml"),
    visualize: Annotated[
        bool,
        typer.Option(
            "--visualize",
            "-v",
            help="Generate and save the pipeline DAG visualization as 'pipeline.png' without running.",
        ),
    ] = False,
):
    """
    Execute the irrigation processing pipeline.
    """

    from irrigation_processor.config import AppConfig
    from irrigation_processor.constants import LOG
    from irrigation_processor.core import LocalService, Pipeline, XcubeDataStoreStorage
    from irrigation_processor.steps import registry

    # current workaround for GDAL env in Dask
    os.environ["AWS_S3_ENDPOINT"] = "eodata.dataspace.copernicus.eu"
    os.environ["AWS_VIRTUAL_HOSTING"] = "FALSE"
    os.environ["AWS_ACCESS_KEY_ID"] = os.environ["CDSE_AWS_ACCESS_KEY_ID"]
    os.environ["AWS_SECRET_ACCESS_KEY"] = os.environ["CDSE_AWS_SECRET_ACCESS_KEY"]

    with open(config, "r") as f:
        raw_config = yaml.safe_load(f)

    storage_config = raw_config.get("storage", {})
    storage = XcubeDataStoreStorage(**storage_config)

    app_config = AppConfig(**raw_config)

    service = LocalService(storage=storage, app_config=app_config)
    p = Pipeline(service=service, pipeline_name=pipeline_name)

    p.add_steps_from_registry(registry)

    if visualize:
        try:
            import graphviz

            dot_str = p.visualize_dot()
            src = graphviz.Source(dot_str)
            output_path = src.render("pipeline", format="png", cleanup=True)
            LOG.info(f"Pipeline visualization saved to: {output_path}")
            return
        except ImportError:
            LOG.error(
                "graphviz library not found. Please install it to use --visualize."
            )
            raise typer.Exit(code=1)

    state = p.run()
    LOG.info(f"State metadata:\n{json.dumps(state, indent=2)}")


@app.command()
def show_version():
    from importlib.metadata import version

    typer.echo(version(APP_NAME))


if __name__ == "__main__":
    app()
