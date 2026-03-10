import os
import unittest
from unittest.mock import Mock, patch

import numpy as np
import xarray as xr

from irrigation_processor.constants import (CLMS_DATA_ID, ERA5_DATA_ID,
                                            LC_DATA_ID)
from irrigation_processor.ops.dataloader import (_get_cds_data, _get_clms_data,
                                                 _get_lc_data, load_data)
from tests.helpers import DummyContext, make_era5_ds
from tests.helpers import make_preprocessed_ds as make_daily_era5_ds
from tests.helpers import make_raw_clms_ds as make_clms_ds


class TestDataLoader(unittest.TestCase):
    def setUp(self):
        self.store = Mock()

        self.store.storage.store_kwargs.root = "output_dir"
        self.context = DummyContext(store=self.store)

    @patch("irrigation_processor.ops.dataloader._get_cds_data")
    @patch("irrigation_processor.ops.dataloader._get_clms_data")
    @patch("irrigation_processor.ops.dataloader._get_lc_data")
    def test_load_data_happy_path(
        self,
        mock_lc,
        mock_clms,
        mock_cds,
    ):
        mock_cds.return_value = ERA5_DATA_ID
        mock_clms.return_value = CLMS_DATA_ID
        mock_lc.return_value = LC_DATA_ID

        result = load_data(self.context, self.store)
        self.assertEqual(result["sm_data_id"], CLMS_DATA_ID)
        self.assertEqual(result["era5_vars_data_id"], ERA5_DATA_ID)
        self.assertEqual(result["lc_data_id"], LC_DATA_ID)
        self.assertEqual(result["gleam_data_id"], None)

    @patch("irrigation_processor.ops.dataloader.split_date_range")
    def test_get_cds_data_cached(self, mock_split):
        self.store.exists.return_value = True

        result = _get_cds_data(self.context, self.store)

        self.assertEqual(result, ERA5_DATA_ID)
        mock_split.assert_not_called()

    @patch("irrigation_processor.ops.dataloader.get_existing_data")
    @patch("irrigation_processor.ops.dataloader.zappend")
    @patch("irrigation_processor.ops.dataloader.new_data_store")
    def test_get_cds_data_non_optimized(
        self,
        mock_new_store,
        mock_zappend,
        mock_get_existing_data,
    ):
        self.context.dataloader.cds_optimize_writing = False

        ds = make_era5_ds()

        mock_get_existing_data.return_value = None
        self.store.list_ids.side_effect = [
            ["era5/part1"],  # after writing chunks
        ]

        cds_store = Mock()
        cds_store.get_data_opener_ids.return_value = ["dataset"]
        cds_store.open_data.return_value = ds
        mock_new_store.return_value = cds_store

        self.store.load.return_value = ds

        result = _get_cds_data(self.context, self.store)

        self.assertEqual(result, ERA5_DATA_ID)

        self.store.save.assert_any_call(
            key="era5/era5-2024_01_01-2024_01_02.zarr",
            obj=ds,
        )

        mock_zappend.assert_called_once()

        slice_source = mock_zappend.call_args.kwargs["slice_source"]
        out_ds = slice_source("era5/part1")

        self.assertEqual(out_ds.sizes["time"], 2)
        self.assertIn("pev", out_ds)
        self.assertIn("tp", out_ds)
        self.assertNotIn("expver", out_ds)
        self.assertNotIn("number", out_ds)

        np.testing.assert_allclose(
            out_ds["pev"].values[0],
            [[-1.0, 2.0], [-3.0, 4.0]],
        )
        np.testing.assert_allclose(
            out_ds["tp"].values[0],
            [[10.0, 20.0], [30.0, 40.0]],
        )

    @patch("irrigation_processor.ops.dataloader.get_existing_data")
    @patch("irrigation_processor.ops.dataloader.zappend")
    @patch("irrigation_processor.ops.dataloader.new_data_store")
    def test_get_cds_data_non_optimized_s3(
        self,
        mock_new_store,
        mock_zappend,
        mock_get_existing_data,
    ):
        self.context.dataloader.cds_optimize_writing = False
        self.store.protocol = "s3"

        ds = make_era5_ds()
        mock_get_existing_data.return_value = None
        self.store.list_ids.side_effect = [
            ["era5/part1"],
        ]

        cds_store = Mock()
        cds_store.get_data_opener_ids.return_value = ["dataset"]
        cds_store.open_data.return_value = ds
        mock_new_store.return_value = cds_store

        self.store.load.return_value = ds
        os.environ["XCUBE_BUCKET_NAME"] = "test-bucket"
        result = _get_cds_data(self.context, self.store)

        self.assertEqual(result, ERA5_DATA_ID)

        mock_zappend.assert_called_once()

        config = mock_zappend.call_args.kwargs["config"]
        self.assertEqual(config["target_dir"], f"s3://test-bucket/{ERA5_DATA_ID}")
        self.assertIn("target_storage_options", config)

        storage_opts = config["target_storage_options"]
        self.assertIn("key", storage_opts)
        self.assertIn("secret", storage_opts)
        self.assertEqual(storage_opts["anon"], False)

    @patch("irrigation_processor.ops.dataloader.get_existing_data")
    @patch("irrigation_processor.ops.dataloader.new_data_store")
    def test_get_cds_data_optimized(
        self,
        mock_new_store,
        mock_get_existing_data,
    ):
        self.context.dataloader.cds_optimize_writing = True

        ds = make_era5_ds()
        mock_get_existing_data.return_value = None
        self.store.list_ids.side_effect = [
            ["era5/part1"],
        ]

        cds_store = Mock()
        cds_store.get_data_opener_ids.return_value = ["dataset"]
        cds_store.open_data.return_value = ds
        mock_new_store.return_value = cds_store

        def open_data_side_effect(data_id):
            if data_id == "era5_chunked.zarr":
                return make_daily_era5_ds(
                    time_range=["2024-01-01"],
                    pev_data=[[[4, 5], [6, 7]]],
                    tp_data=[[[40, 50], [60, 70]]],
                )
            return ds

        self.store.load.side_effect = open_data_side_effect

        result = _get_cds_data(self.context, self.store)

        written_ds = self.store.save.call_args_list[-1][1]["obj"]

        self.assertEqual(written_ds.sizes["time"], 1)
        self.assertFalse(np.allclose(written_ds["pev"].values[0], ds["pev"].values[0]))
        np.testing.assert_allclose(
            written_ds["pev"].values[0],
            [[4, 5], [6, 7]],
        )
        np.testing.assert_allclose(
            written_ds["tp"].values[0],
            [[40, 50], [60, 70]],
        )

        self.assertEqual(result, ERA5_DATA_ID)
        self.store.save.assert_any_call(
            key="era5/era5-2024_01_01-2024_01_02.zarr", obj=ds
        )

        calls = self.store.save.call_args_list
        chunked_calls = [
            call for call in calls if call[1]["key"] == "era5_chunked.zarr"
        ]
        self.assertEqual(len(chunked_calls), 1)
        chunked_ds = chunked_calls[0][1]["obj"]
        self.assertEqual(chunked_ds.sizes["time"], 2)
        self.assertIn("pev", chunked_ds)
        self.assertIn("tp", chunked_ds)
        self.assertNotIn("expver", chunked_ds)
        self.assertNotIn("number", chunked_ds)

        final_calls = [call for call in calls if call[1]["key"] == ERA5_DATA_ID]
        self.assertEqual(len(final_calls), 1)
        final_ds = final_calls[0][1]["obj"]
        self.assertEqual(final_ds.sizes["time"], 1)

        self.store.delete.assert_called_once_with("era5_chunked.zarr")

    @patch("irrigation_processor.ops.dataloader.open", create=True)
    def test_get_clms_data_cached(self, mock_open):
        self.store.exists.return_value = True

        result = _get_clms_data(self.context, self.store)

        self.assertEqual(result, CLMS_DATA_ID)

    @patch("irrigation_processor.ops.dataloader.get_existing_data")
    @patch("irrigation_processor.ops.dataloader.time.sleep")
    @patch("irrigation_processor.ops.dataloader.zappend")
    @patch("irrigation_processor.ops.dataloader.new_data_store")
    @patch("irrigation_processor.ops.dataloader.open", create=True)
    def test_get_clms_data_non_optimized(
        self,
        mock_open,
        mock_new_store,
        mock_zappend,
        mock_sleep,
        mock_get_existing_data,
    ):
        ds = make_clms_ds()

        mock_get_existing_data.return_value = None
        self.store.list_ids.side_effect = [
            ["clms/part1"],  # after writing chunks
        ]

        mock_open.return_value.__enter__.return_value.read.return_value = "{}"

        clms_store = Mock()
        clms_store.open_data.return_value = ds
        mock_new_store.return_value = clms_store

        self.store.load.return_value = ds

        result = _get_clms_data(self.context, self.store)

        self.assertEqual(result, CLMS_DATA_ID)

        self.store.save.assert_any_call(
            key="clms/clms_sm-2024_01_01-2024_01_02.zarr",
            obj=ds.rename({"x": "lon", "y": "lat"}),
        )

        mock_zappend.assert_called_once()

        slice_source = mock_zappend.call_args.kwargs["slice_source"]
        out_ds = slice_source("clms/part1")

        self.assertIn("ssm", out_ds)
        self.assertEqual(out_ds["ssm"].dtype, np.float32)
        self.assertEqual(out_ds.sizes["time"], 2)

    @patch("irrigation_processor.ops.dataloader.get_existing_data")
    @patch("irrigation_processor.ops.dataloader.time.sleep")
    @patch("irrigation_processor.ops.dataloader.zappend")
    @patch("irrigation_processor.ops.dataloader.new_data_store")
    @patch("irrigation_processor.ops.dataloader.open", create=True)
    def test_get_clms_data_non_optimized_s3(
        self,
        mock_open,
        mock_new_store,
        mock_zappend,
        mock_sleep,
        mock_get_existing_data,
    ):
        self.store.protocol = "s3"
        ds = make_clms_ds()

        mock_get_existing_data.return_value = None
        self.store.list_ids.side_effect = [
            ["clms/part1"],
        ]

        mock_open.return_value.__enter__.return_value.read.return_value = "{}"

        clms_store = Mock()
        clms_store.open_data.return_value = ds
        mock_new_store.return_value = clms_store

        self.store.load.return_value = ds

        os.environ["XCUBE_BUCKET_NAME"] = "test-bucket"
        result = _get_clms_data(self.context, self.store)

        self.assertEqual(result, CLMS_DATA_ID)
        mock_zappend.assert_called_once()

        config = mock_zappend.call_args.kwargs["config"]

        self.assertEqual(config["target_dir"], f"s3://test-bucket/{CLMS_DATA_ID}")
        self.assertIn("target_storage_options", config)

        storage_opts = config["target_storage_options"]
        self.assertEqual(storage_opts["anon"], False)
        self.assertIn("key", storage_opts)
        self.assertIn("secret", storage_opts)

    @patch("irrigation_processor.ops.dataloader.get_existing_data")
    def test_get_lc_data_cached(self, mock_get_existing_data):
        mock_get_existing_data.return_value = "lc_data_id"

        result = _get_lc_data(self.context, self.store)
        self.assertIs(result, "lc_data_id")

    @patch("irrigation_processor.ops.dataloader.new_data_store")
    def test_get_lc_data_download(self, mock_new_store):
        self.store.exists.return_value = False

        base_ds = xr.Dataset(
            {
                "crs": ((), 1),
                "lccs_class": (("time",), [2]),
            },
            coords={"time": ["2020"]},
        )

        lc_store = Mock()
        lc_store.open_data.return_value = base_ds
        mock_new_store.return_value = lc_store

        result = _get_lc_data(self.context, self.store)

        self.assertIsInstance(result, str)
