import os
import unittest
from unittest.mock import Mock, patch

import pandas as pd
import numpy as np
import xarray as xr
from pydantic import BaseModel, ConfigDict

from irrigation_processor.constants import (
    CLMS_DATA_ID,
    ERA5_DATA_ID,
    LC_DATA_ID,
)
from irrigation_processor.ops.dataloader import (
    load_data,
    _get_cds_data,
    _get_clms_data,
    _get_lc_data,
)


class DummyContext(BaseModel):
    store: Mock
    time_range: tuple = ("2020-01-01", "2020-01-02")
    bbox: list = [0, 0, 1, 1]
    cds_data_id: str = "era5"
    cds_spatial_res: float = 0.1
    cds_variable_names: list = ["pev", "tp"]
    cds_optimize_writing: bool = False
    lc_time: str = "2020"

    model_config = ConfigDict(arbitrary_types_allowed=True)


def make_era5_ds():
    time = pd.date_range("2020-01-01", periods=4, freq="6H")

    pev = np.array(
        [
            [[1, 2], [3, 4]],
            [[2, 3], [4, 5]],
            [[3, 4], [5, 6]],
            [[4, 5], [6, 7]],
        ]
    )

    tp = np.array(
        [
            [[10, 20], [30, 40]],
            [[20, 30], [40, 50]],
            [[30, 40], [50, 60]],
            [[40, 50], [60, 70]],
        ]
    )

    return xr.Dataset(
        {
            "pev": (("time", "lat", "lon"), pev),
            "tp": (("time", "lat", "lon"), tp),
            "expver": ("time", [1, 1, 1, 1]),
            "number": ("time", [0, 0, 0, 0]),
        },
        coords={
            "time": time,
            "lat": [5, 4],
            "lon": [5, 6],
        },
    )


EXPECTED_PEV = np.array(
    [
        [2.5, 3.5],
        [4.5, 5.5],
    ]
)

EXPECTED_TP = np.array(
    [
        [25.0, 35.0],
        [45.0, 55.0],
    ]
)


def make_daily_era5_ds():
    return xr.Dataset(
        {
            "pev": (("time", "lat", "lon"), [EXPECTED_PEV]),
            "tp": (("time", "lat", "lon"), [EXPECTED_TP]),
        },
        coords={
            "time": ["2020-01-01"],
            "lat": [5, 4],
            "lon": [5, 6],
        },
    )


def make_clms_ds():
    time = pd.date_range("2020-01-01", periods=2, freq="1D")

    ssm = np.array(
        [
            [[0.10, 0.20], [0.30, 0.40]],
            [[0.20, 0.30], [0.40, 0.50]],
        ]
    )

    ssm_noise = np.array(
        [
            [[1, 1], [1, 1]],
            [[1, 1], [1, 1]],
        ]
    )

    return xr.Dataset(
        {
            "ssm": (("time", "y", "x"), ssm),
            "ssm_noise": (("time", "y", "x"), ssm_noise),
        },
        coords={
            "time": time,
            "y": [5, 4],
            "x": [5, 6],
        },
    )


class TestDataLoader(unittest.TestCase):
    def setUp(self):
        self.store = Mock()
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
        mock_lc.return_value = xr.Dataset()

        result = load_data(self.context)

        self.assertEqual(result["sm_data_id"], CLMS_DATA_ID)
        self.assertEqual(result["era5_data_id"], ERA5_DATA_ID)
        self.assertIsInstance(result[LC_DATA_ID], xr.Dataset)

    @patch("irrigation_processor.ops.dataloader.split_date_range")
    def test_get_cds_data_cached(self, mock_split):
        self.store.list_data_ids.return_value = [ERA5_DATA_ID]

        result = _get_cds_data(self.context)

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
        self.context.cds_optimize_writing = False

        ds = make_era5_ds()

        mock_get_existing_data.return_value = None
        self.store.list_data_ids.side_effect = [
            ["era5/part1"],  # after writing chunks
        ]

        cds_store = Mock()
        cds_store.get_data_opener_ids.return_value = ["dataset"]
        cds_store.open_data.return_value = ds
        mock_new_store.return_value = cds_store

        self.store.open_data.return_value = ds

        result = _get_cds_data(self.context)

        self.assertEqual(result, ERA5_DATA_ID)

        self.store.write_data.assert_any_call(
            ds,
            "era5/era5-2020_01_01-2020_01_02.zarr",
            replace=True,
        )

        mock_zappend.assert_called_once()

        slice_source = mock_zappend.call_args.kwargs["slice_source"]
        out_ds = slice_source("era5/part1")

        self.assertEqual(out_ds.sizes["time"], 1)
        self.assertIn("pev", out_ds)
        self.assertIn("tp", out_ds)
        self.assertNotIn("expver", out_ds)
        self.assertNotIn("number", out_ds)

        np.testing.assert_allclose(
            out_ds["pev"].values[0],
            EXPECTED_PEV,
        )
        np.testing.assert_allclose(
            out_ds["tp"].values[0],
            EXPECTED_TP,
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
        self.context.cds_optimize_writing = False
        self.store.protocol = "s3"

        ds = make_era5_ds()
        mock_get_existing_data.return_value = None
        self.store.list_data_ids.side_effect = [
            ["era5/part1"],
        ]

        cds_store = Mock()
        cds_store.get_data_opener_ids.return_value = ["dataset"]
        cds_store.open_data.return_value = ds
        mock_new_store.return_value = cds_store

        self.store.open_data.return_value = ds
        os.environ["XCUBE_BUCKET_NAME"] = "test-bucket"
        result = _get_cds_data(self.context)

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
        self.context.cds_optimize_writing = True

        ds = make_era5_ds()
        mock_get_existing_data.return_value = None
        self.store.list_data_ids.side_effect = [
            ["era5/part1"],
        ]

        cds_store = Mock()
        cds_store.get_data_opener_ids.return_value = ["dataset"]
        cds_store.open_data.return_value = ds
        mock_new_store.return_value = cds_store

        def open_data_side_effect(data_id):
            if data_id == "era5_chunked.zarr":
                return make_daily_era5_ds()
            return ds

        self.store.open_data.side_effect = open_data_side_effect
        # self.store.open_data.return_value = ds

        result = _get_cds_data(self.context)

        written_ds = self.store.write_data.call_args_list[-1][0][0]

        self.assertEqual(written_ds.sizes["time"], 1)
        self.assertFalse(np.allclose(written_ds["pev"].values[0], ds["pev"].values[0]))
        np.testing.assert_allclose(
            written_ds["pev"].values[0],
            EXPECTED_PEV,
        )
        np.testing.assert_allclose(
            written_ds["tp"].values[0],
            EXPECTED_TP,
        )

        self.assertEqual(result, ERA5_DATA_ID)
        self.store.write_data.assert_any_call(
            ds, "era5/era5-2020_01_01-2020_01_02.zarr", replace=True
        )

        calls = self.store.write_data.call_args_list
        chunked_calls = [call for call in calls if call[0][1] == "era5_chunked.zarr"]
        self.assertEqual(len(chunked_calls), 1)
        chunked_ds = chunked_calls[0][0][0]
        self.assertEqual(chunked_ds.sizes["time"], 1)
        self.assertIn("pev", chunked_ds)
        self.assertIn("tp", chunked_ds)
        self.assertNotIn("expver", chunked_ds)
        self.assertNotIn("number", chunked_ds)

        final_calls = [call for call in calls if call[0][1] == ERA5_DATA_ID]
        self.assertEqual(len(final_calls), 1)
        final_ds = final_calls[0][0][0]
        self.assertEqual(final_ds.sizes["time"], 1)

        self.store.delete_data.assert_called_once_with("era5_chunked.zarr")

    @patch("irrigation_processor.ops.dataloader.open", create=True)
    def test_get_clms_data_cached(self, mock_open):
        self.store.list_data_ids.return_value = [CLMS_DATA_ID]

        result = _get_clms_data(self.context)

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
        self.store.list_data_ids.side_effect = [
            ["clms/part1"],  # after writing chunks
        ]

        mock_open.return_value.__enter__.return_value.read.return_value = "{}"

        clms_store = Mock()
        clms_store.open_data.return_value = ds
        mock_new_store.return_value = clms_store

        self.store.open_data.return_value = ds

        result = _get_clms_data(self.context)

        self.assertEqual(result, CLMS_DATA_ID)

        self.store.write_data.assert_any_call(
            ds.rename({"x": "lon", "y": "lat"}),
            "clms/clms_sm-2020_01_01-2020_01_02.zarr",
            replace=True,
        )

        mock_zappend.assert_called_once()

        slice_source = mock_zappend.call_args.kwargs["slice_source"]
        out_ds = slice_source("clms/part1")

        self.assertIn("ssm", out_ds)
        self.assertNotIn("ssm_noise", out_ds)
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
        self.store.list_data_ids.side_effect = [
            ["clms/part1"],
        ]

        mock_open.return_value.__enter__.return_value.read.return_value = "{}"

        clms_store = Mock()
        clms_store.open_data.return_value = ds
        mock_new_store.return_value = clms_store

        self.store.open_data.return_value = ds

        os.environ["XCUBE_BUCKET_NAME"] = "test-bucket"
        result = _get_clms_data(self.context)

        self.assertEqual(result, CLMS_DATA_ID)
        mock_zappend.assert_called_once()

        config = mock_zappend.call_args.kwargs["config"]

        self.assertEqual(config["target_dir"], f"s3://test-bucket"
                                               f"/{CLMS_DATA_ID}")
        self.assertIn("target_storage_options", config)

        storage_opts = config["target_storage_options"]
        self.assertEqual(storage_opts["anon"], False)
        self.assertIn("key", storage_opts)
        self.assertIn("secret", storage_opts)

    @patch("irrigation_processor.ops.dataloader.get_existing_data")
    def test_get_lc_data_cached(self, mock_get_existing_data):
        ds = xr.Dataset()
        mock_get_existing_data.return_value = ds

        result = _get_lc_data(self.context)
        self.assertIs(result, ds)

    @patch("irrigation_processor.ops.dataloader.new_data_store")
    def test_get_lc_data_download(self, mock_new_store):
        self.store.list_data_ids.return_value = []

        base_ds = xr.Dataset(
            {
                "crs": ((), 1),
                "lccs_class": (("time",), [2]),
            },
            coords={"time": ["2020"]},
        )

        lc_store = Mock()
        lc_store.open_data.return_value.base_dataset = base_ds
        mock_new_store.return_value = lc_store

        result = _get_lc_data(self.context)

        self.assertIsInstance(result, xr.Dataset)
        self.assertIn("lccs_class", result)
