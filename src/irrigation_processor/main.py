import json

from src.irrigation_processor.pipeline import (
    InlineService,
    FileStorage,
    Pipeline,
)

from src.irrigation_processor.steps import step

registry = step.get_registry()

storage = FileStorage("./pipeline_storage")
service = InlineService(storage=storage)
p = Pipeline(service=service)

p.add_steps_from_registry(registry)
state = p.run()
print("State metadata:\n", json.dumps(state, indent=2))
