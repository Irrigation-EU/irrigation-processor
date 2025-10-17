import typer

from irrigation_processor.pipeline import StepRegistry
from irrigation_processor.pipeline_framework.util import AliasedGroup

STEP_REGISTRY_GETTER_KEY = "get_step_registry"

CLI_HELP = """Command-line interface for pipeline steps description and 
execution.

You can use shorter command name aliases, e.g., use command name `ep`
for `execute-pipeline`, or `lps` for `list-pipeline-steps`.
"""


cli = typer.Typer(
    add_completion=False,
    cls=AliasedGroup,
    help=CLI_HELP,
    context_settings={
        "obj": {STEP_REGISTRY_GETTER_KEY: None},
    },
)


def get_cli(
    step_registry: StepRegistry,
    **kwargs,
) -> typer.Typer:
    assert callable(step_registry)
    context_settings = cli.info.context_settings
    assert isinstance(context_settings, dict)
    context_obj = context_settings["obj"]
    assert isinstance(context_obj, dict)
    context_obj.update({STEP_REGISTRY_GETTER_KEY: step_registry, **kwargs})
    return cli


@cli.command("execute-pipeline")
def execute_pipeline(
    ctx: typer.Context,
):
    """Execute a pipeline."""

