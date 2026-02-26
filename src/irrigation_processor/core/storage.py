import abc
from typing import Any

import xarray as xr
from xcube.core.store import new_data_store

from irrigation_processor.constants import LOG


class Storage(abc.ABC):
    """
    Abstract interface for storing and retrieving pipeline data.

    A storage backend is responsible for saving step outputs, loading them
    again when needed, and other abstract methods defined below.
    """

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
    """
    Storage implementation backed by xcube data store.

    Designed for storing and loading xarray Datasets.
    """

    def __init__(self, store_id: str = "file", store_kwargs: dict | None = None):
        if not store_kwargs:
            store_kwargs = {}
        if store_id == "file" and "root" not in store_kwargs:
            store_kwargs.update({"root": "output_irrigation", "max_depth": 5})
        self._store_id = store_id
        self._store = new_data_store(store_id, **store_kwargs)

    def save(self, key: str, obj: Any) -> str:
        """
        Save a xarray Dataset to the data store.

        Args:
            key: Identifier to use for storing the dataset.
            obj: The xarray.Dataset to store.

        Returns:
            The data identifier used to store the dataset.

        Raises:
            TypeError: If obj is not an xarray.Dataset.
        """
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
        """
        Load a dataset from the data store.

        Args:
            data_id: Identifier of the stored dataset.

        Returns:
            The loaded xarray.Dataset.
        """
        return self._store.open_data(data_id)

    def exists(self, data_id: str) -> bool:
        """
        Check whether a dataset exists in the store.

        Args:
            data_id: Identifier of the stored dataset.

        Returns:
            True if the dataset exists, otherwise False.
        """
        return self._store.has_data(data_id)

    def list_ids(self) -> list[str]:
        """
        List all dataset identifiers in the store.

        Returns:
            A list of available data identifiers.
        """
        return list(self._store.list_data_ids())

    def delete(self, data_id: str) -> None:
        """
        Delete a dataset from the store.

        Args:
            data_id: Identifier of the dataset to delete.
        """
        self._store.delete_data(data_id)

    @property
    def protocol(self) -> str:
        """
        Return the underlying storage protocol.

        Returns:
            A string representing the storage protocol
            (e.g., "file", "s3").
        """
        return self._store.protocol
