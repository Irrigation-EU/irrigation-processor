from dask.distributed import Client, LocalCluster
import json

from src.irrigation_processor.pipeline import FileStorage, LocalService, Pipeline
from src.irrigation_processor.steps import step


def main():
    registry = step.get_registry()

    # TODO:  Use Storage for xcube data store

    storage = FileStorage("./pipeline_storage")
    service = LocalService(storage=storage)
    p = Pipeline(service=service)

    p.add_steps_from_registry(registry)
    state = p.run()
    print("State metadata:\n", json.dumps(state, indent=2))

if __name__ == "__main__":
    cluster = LocalCluster(n_workers=4, threads_per_worker=2)
    client = Client(cluster)

    main()
