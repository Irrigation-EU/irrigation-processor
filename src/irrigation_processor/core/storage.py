import abc
from typing import Any

import xarray as xr
from xcube.core.store import new_data_store

from irrigation_processor.constants import LOG


class Storage(abc.ABC):
    @abc.abstractmethod
    def save(self, key: str, obj: Any) -> str:
        """
        Persist an object and return its data_id.
        """

    @abc.abstractmethod
    def load(self, data_id: str) -> Any:
        """
        Retrieve an object by its data_id.
        """

    @abc.abstractmethod
    def exists(self, data_id: str) -> bool:
        """Return True if the object exists."""

    @abc.abstractmethod
    def list_ids(self) -> list[str]:
        """
        Return all stored data identifiers known to this backend.
        """

    @abc.abstractmethod
    def delete(self, data_id: str) -> None:
        """
        Delete the stored object referenced by data_id.
        """

    @property
    @abc.abstractmethod
    def protocol(self) -> str:
        """
        Return the storage protocol (e.g., 'file', 's3').
        """


class XcubeDataStoreStorage(Storage):
    def __init__(self, store_id: str = "file", store_kwargs: dict | None = None):
        if not store_kwargs:
            store_kwargs = {}
        if store_id == "file" and "root" not in store_kwargs:
            store_kwargs.update({"root": "output_irrigation", "max_depth": 5})
        self._store_id = store_id
        self._store = new_data_store(store_id, **store_kwargs)

    def save(self, key: str, obj: Any) -> str:
        if not isinstance(obj, xr.Dataset):
            raise TypeError(
                f"XcubeDataStoreStorage only supports xr.Dataset, got {type(obj)}"
            )

        data_id = key
        data_ids = self._store.list_data_ids()

        if data_id not in data_ids:
            LOG.info(
                f"Data id {data_id} does not exist in the xcube data store. Writing to it."
            )
            self._store.write_data(obj, data_id, replace=False)
        else:
            LOG.info(
                f"Data id {data_id} already exists in the xcube data store. Using cached data."
            )

        return data_id

    def load(self, data_id: str) -> Any:
        return self._store.open_data(data_id)

    def exists(self, data_id: str) -> bool:
        return self._store.has_data(data_id)

    def list_ids(self) -> list[str]:
        return list(self._store.list_data_ids())

    def delete(self, data_id: str) -> None:
        self._store.delete_data(data_id)

    @property
    def protocol(self) -> str:
        return self._store.protocol
