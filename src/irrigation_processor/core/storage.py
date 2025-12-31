import abc
from typing import Any

import xarray as xr
from xcube.core.store import new_data_store

from irrigation_processor.constants import LOG, OUTPUT_DIR


class Storage(abc.ABC):
    @abc.abstractmethod
    def save(self, key: str, obj: Any) -> dict[str, Any]:
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
    def load(self, metadata: dict[str, Any]) -> Any:
        """Load an object previously saved using the metadata returned by save.

        This method must handle the loading of the 3 cases as discussed in
        save() method.
        """


class XcubeDataStoreStorage(Storage):
    def __init__(self, store_id: str = "file", store_kwargs: dict | None = None):
        if not store_kwargs:
            store_kwargs = {}
        if store_id == "file" and "root" not in store_kwargs:
            store_kwargs.update({"root": OUTPUT_DIR, "max_depth": 5})
        self.store = new_data_store(store_id, **store_kwargs)

    def save(self, key: str, obj: Any) -> dict[str, Any]:
        if isinstance(obj, (int, float, str, bool)):
            return {"inline": True, "value": obj, "type": type(obj).__name__}

        if isinstance(obj, xr.Dataset):
            data_id = key
            data_ids = self.store.list_data_ids()

            if data_id in data_ids:
                LOG.info(
                    f"Data id {data_id} already exists in the xcube data store. Using cached data."
                )
                return {"inline": False, "data_id": data_id, "type": type(obj).__name__}
            LOG.info(
                f"Data id {data_id} does not exist in the xcube data store. Writing to it."
            )
            self.store.write_data(obj, data_id)
            return {"inline": False, "data_id": data_id, "type": type(obj).__name__}

        raise RuntimeError(f"Unknown storage format: {type(obj)}")

    def load(self, metadata: dict[str, Any]) -> Any:
        if metadata.get("inline"):
            return metadata["value"]

        data_id = metadata.get("data_id")
        if not data_id:
            raise RuntimeError(f"Invalid data_id: {data_id}")

        return self.store.open_data(data_id)
