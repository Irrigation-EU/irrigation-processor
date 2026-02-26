import unittest
from unittest.mock import Mock, patch

import xarray as xr

from irrigation_processor.core.storage import Storage, XcubeDataStoreStorage


class TestStorage(unittest.TestCase):
    def test_storage_is_abstract(self):
        with self.assertRaises(TypeError):
            Storage()

    @patch("irrigation_processor.core.storage.new_data_store")
    def test_init_file_store_sets_default_root(self, mock_new_store):
        mock_new_store.return_value = Mock()

        storage = XcubeDataStoreStorage(store_id="file")

        mock_new_store.assert_called_once()
        self.assertIsNotNone(storage._store)

    @patch("irrigation_processor.core.storage.new_data_store")
    def test_init_custom_store_kwargs(self, mock_new_store):
        mock_new_store.return_value = Mock()

        storage = XcubeDataStoreStorage(
            store_id="s3",
            store_kwargs={"a": 1},
        )

        mock_new_store.assert_called_once_with("s3", a=1)
        self.assertIsNotNone(storage._store)

    @patch("irrigation_processor.core.storage.new_data_store")
    def test_save_dataset_cached(self, mock_new_store):
        store = Mock()
        store.list_data_ids.return_value = ["data1"]
        mock_new_store.return_value = store

        storage = XcubeDataStoreStorage()
        ds = xr.Dataset()

        data_id = storage.save("data1", ds)

        self.assertEqual(data_id, "data1")
        # Should not write if already exists
        store.write_data.assert_not_called()

    @patch("irrigation_processor.core.storage.new_data_store")
    def test_save_dataset_new(self, mock_new_store):
        store = Mock()
        store.list_data_ids.return_value = []
        mock_new_store.return_value = store

        storage = XcubeDataStoreStorage()
        ds = xr.Dataset()

        data_id = storage.save("data2", ds)

        store.write_data.assert_called_once_with(ds, "data2", replace=False)
        self.assertEqual(data_id, "data2")

    @patch("irrigation_processor.core.storage.new_data_store")
    def test_save_unknown_type_raises(self, mock_new_store):
        mock_new_store.return_value = Mock()
        storage = XcubeDataStoreStorage()

        with self.assertRaises(TypeError) as ctx:
            storage.save("key", object())

        self.assertIn("only supports xr.Dataset", str(ctx.exception))

    @patch("irrigation_processor.core.storage.new_data_store")
    def test_load_dataset(self, mock_new_store):
        store = Mock()
        store.open_data.return_value = "DATASET"
        mock_new_store.return_value = store

        storage = XcubeDataStoreStorage()

        result = storage.load("data1")

        store.open_data.assert_called_once_with("data1")
        self.assertEqual(result, "DATASET")

    @patch("irrigation_processor.core.storage.new_data_store")
    def test_exists(self, mock_new_store):
        store = Mock()
        store.has_data.return_value = True
        mock_new_store.return_value = store

        storage = XcubeDataStoreStorage()
        self.assertTrue(storage.exists("data1"))
        store.has_data.assert_called_once_with("data1")

    @patch("irrigation_processor.core.storage.new_data_store")
    def test_list_ids(self, mock_new_store):
        store = Mock()
        store.list_data_ids.return_value = ["a", "b"]
        mock_new_store.return_value = store

        storage = XcubeDataStoreStorage()
        self.assertEqual(storage.list_ids(), ["a", "b"])

    @patch("irrigation_processor.core.storage.new_data_store")
    def test_delete(self, mock_new_store):
        store = Mock()
        mock_new_store.return_value = store

        storage = XcubeDataStoreStorage()
        storage.delete("data1")
        store.delete_data.assert_called_once_with("data1")

