import unittest
from unittest.mock import Mock, patch

import xarray as xr

from irrigation_processor.core.storage import (
    Storage,
    XcubeDataStoreStorage,
)


class TestStorage(unittest.TestCase):
    def test_storage_is_abstract(self):
        with self.assertRaises(TypeError):
            Storage()

    @patch("irrigation_processor.core.storage.new_data_store")
    def test_init_file_store_sets_default_root(self, mock_new_store):
        mock_new_store.return_value = Mock()

        storage = XcubeDataStoreStorage(store_id="file")

        mock_new_store.assert_called_once()
        self.assertIsNotNone(storage.store)

    @patch("irrigation_processor.core.storage.new_data_store")
    def test_init_custom_store_kwargs(self, mock_new_store):
        mock_new_store.return_value = Mock()

        storage = XcubeDataStoreStorage(
            store_id="s3",
            store_kwargs={"a": 1},
        )

        mock_new_store.assert_called_once_with("s3", a=1)
        self.assertIsNotNone(storage.store)

    @patch("irrigation_processor.core.storage.new_data_store")
    def test_save_inline_types(self, mock_new_store):
        mock_new_store.return_value = Mock()

        storage = XcubeDataStoreStorage()

        for value in [1, 1.5, "x", True]:
            meta = storage.save("key", value)

            self.assertEqual(meta["inline"], True)
            self.assertEqual(meta["value"], value)
            self.assertEqual(meta["type"], type(value).__name__)

    @patch("irrigation_processor.core.storage.new_data_store")
    def test_save_dataset_cached(self, mock_new_store):
        store = Mock()
        store.list_data_ids.return_value = ["data1"]
        mock_new_store.return_value = store

        storage = XcubeDataStoreStorage()
        ds = xr.Dataset()

        meta = storage.save("data1", ds)

        self.assertEqual(
            meta,
            {"inline": False, "data_id": "data1", "type": "Dataset"},
        )
        store.write_data.assert_not_called()

    @patch("irrigation_processor.core.storage.new_data_store")
    def test_save_dataset_new(self, mock_new_store):
        store = Mock()
        store.list_data_ids.return_value = []
        mock_new_store.return_value = store

        storage = XcubeDataStoreStorage()
        ds = xr.Dataset()

        meta = storage.save("data2", ds)

        store.write_data.assert_called_once_with(ds, "data2")
        self.assertEqual(
            meta,
            {"inline": False, "data_id": "data2", "type": "Dataset"},
        )

    @patch("irrigation_processor.core.storage.new_data_store")
    def test_save_unknown_type_raises(self, mock_new_store):
        mock_new_store.return_value = Mock()
        storage = XcubeDataStoreStorage()

        with self.assertRaises(RuntimeError) as ctx:
            storage.save("key", object())

        self.assertIn("Unknown storage format", str(ctx.exception))

    @patch("irrigation_processor.core.storage.new_data_store")
    def test_load_inline_value(self, mock_new_store):
        mock_new_store.return_value = Mock()
        storage = XcubeDataStoreStorage()

        value = storage.load({"inline": True, "value": 123})

        self.assertEqual(value, 123)

    @patch("irrigation_processor.core.storage.new_data_store")
    def test_load_dataset(self, mock_new_store):
        store = Mock()
        store.open_data.return_value = "DATASET"
        mock_new_store.return_value = store

        storage = XcubeDataStoreStorage()

        result = storage.load({"inline": False, "data_id": "data1"})

        store.open_data.assert_called_once_with("data1")
        self.assertEqual(result, "DATASET")

    @patch("irrigation_processor.core.storage.new_data_store")
    def test_load_missing_data_id_raises(self, mock_new_store):
        mock_new_store.return_value = Mock()
        storage = XcubeDataStoreStorage()

        with self.assertRaises(RuntimeError) as ctx:
            storage.load({"inline": False})

        self.assertIn("Invalid data_id", str(ctx.exception))
