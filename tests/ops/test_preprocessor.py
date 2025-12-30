import numpy as np
import pandas as pd
import xarray as xr

import unittest
from unittest.mock import Mock, patch

from xcube_resampling.gridmapping import GridMapping

from irrigation_processor.ops.preprocessor import (
    irrigation_preprocessor,
    _soil_moisture_preprocessor,
    _land_cover_preprocessor,
    _era5_preprocessor,
    _swicomp_nan,
    _resample_and_merge,
)
from irrigation_processor.constants import (
    INPUT_FOR_CALIBRATION_ID,
    PROCESSED_CLMS_DATA_ID,
)


def make_clms_ds():
    time = pd.date_range("2020-01-01", periods=3, freq="D")
    return xr.Dataset(
        {
            "ssm": (
                ("time", "lat", "lon"),
                [[[10, 10], [30, 90]], [[20, 40], [10, 60]], [[300, 40], [500, 60]]],
            ),
        },
        coords={
            "time": time,
            "lat": [5, 4],
            "lon": [5, 6],
        },
    )


def make_clms_ds_with_gap():
    time = pd.to_datetime(["2020-01-01", "2020-01-03"])

    return xr.Dataset(
        {
            "ssm": (
                ("time", "lat", "lon"),
                [
                    [[10, 10], [30, 90]],
                    [[300, 40], [500, 60]],
                ],
            ),
        },
        coords={
            "time": time,
            "lat": [5, 4],
            "lon": [5, 6],
        },
    )


def make_lc_ds():
    return xr.Dataset(
        {"lccs_class": (("lat", "lon"), [[10, 30], [40, 11]])},
        coords={
            "lat": [5, 4],
            "lon": [5, 6],
        },
    )


def make_era5_ds():
    time = pd.date_range("2020-01-01", periods=2, freq="D")
    return xr.Dataset(
        {
            "pev": (("time", "lat", "lon"), [[[-1, 2], [-3, 4]], [[2, -3], [4, 5]]]),
            "tp": (
                ("time", "lat", "lon"),
                [[[10, 20], [30, 40]], [[20, 30], [40, 50]]],
            ),
        },
        coords={
            "time": time,
            "lat": [5, 4],
            "lon": [5, 6],
        },
    )


def make_swi_ds():
    time = pd.date_range("2020-01-01", periods=3, freq="D")
    return xr.Dataset(
        {
            "SWI": (
                ("time", "lat", "lon"),
                [
                    [[0.1, 0.2], [0.3, 0.4]],
                    [[0.2, 0.3], [0.4, 0.5]],
                    [[0.3, 0.4], [0.5, 0.6]],
                ],
            ),
        },
        coords={
            "time": time,
            "lat": [4, 5],
            "lon": [5, 4],
        },
    )


def make_lc_mask():
    return xr.DataArray(
        [[1, 0], [1, 1]],
        dims=("lat", "lon"),
        coords={
            "time": pd.to_datetime("2020-01-01"),
            "lat": [4, 5],
            "lon": [5, 4],
        },
    )


def make_era5_daily_ds():
    time = pd.date_range("2020-01-01", periods=3, freq="D")
    return xr.Dataset(
        {
            "pev": (
                ("time", "lat", "lon"),
                [[[1, 2], [3, 4]], [[2, 3], [4, 5]], [[3, 4], [5, 6]]],
            ),
            "tp": (
                ("time", "lat", "lon"),
                [[[10, 20], [30, 40]], [[20, 30], [40, 50]], [[30, 40], [50, 60]]],
            ),
        },
        coords={
            "time": time,
            "lat": [4, 5],
            "lon": [5, 4],
        },
    )


class DummyContext:
    def __init__(self, store):
        self.store = store
        self.bbox = (5, 4, 6, 5)  # xmin, ymin, xmax, ymax


class TestPreprocessor(unittest.TestCase):
    def test_irrigation_preprocessor_cached(self):
        store = Mock()
        store.list_data_ids.return_value = [INPUT_FOR_CALIBRATION_ID]
        store.open_data.return_value = "CACHED_DS"

        ctx = DummyContext(store)

        out = irrigation_preprocessor(ctx, "sm", "lc", "era5")

        self.assertEqual(out, "CACHED_DS")

    @patch("irrigation_processor.ops.preprocessor._resample_and_merge")
    @patch("irrigation_processor.ops.preprocessor._era5_preprocessor")
    @patch("irrigation_processor.ops.preprocessor._land_cover_preprocessor")
    @patch("irrigation_processor.ops.preprocessor._soil_moisture_preprocessor")
    def test_irrigation_preprocessor_flow(
        self,
        mock_sm,
        mock_lc,
        mock_era5,
        mock_merge,
    ):
        store = Mock()
        store.list_data_ids.return_value = []

        ctx = DummyContext(store)

        mock_sm.return_value = "SM"
        mock_lc.return_value = "LC"
        mock_era5.return_value = "ERA5"
        mock_merge.return_value = "MERGED"

        out = irrigation_preprocessor(ctx, "sm", "lc", "era5")

        self.assertEqual(out, "MERGED")
        mock_merge.assert_called_once_with("SM", "LC", "ERA5")

    def test_soil_moisture_preprocessor_cached(self):
        store = Mock()
        store.list_data_ids.return_value = [PROCESSED_CLMS_DATA_ID]
        store.open_data.return_value = "CACHED_SWI"

        ctx = DummyContext(store)

        out = _soil_moisture_preprocessor(ctx, "sm")

        self.assertEqual(out, "CACHED_SWI")

    def test_soil_moisture_preprocessor(self):
        store = Mock()
        store.list_data_ids.return_value = []
        store.open_data.return_value = make_clms_ds()

        ctx = DummyContext(store)

        out = _soil_moisture_preprocessor(ctx, "sm")

        self.assertIn("SWI", out)
        self.assertEqual(out["SWI"].dims, ("time", "lat", "lon"))
        self.assertEqual(out["SWI"].sizes, {"time": 3, "lat": 2, "lon": 2})

    def test_soil_moisture_preprocessor_fills_missing_dates(self):
        store = Mock()
        store.list_data_ids.return_value = []
        store.open_data.return_value = make_clms_ds_with_gap()

        ctx = DummyContext(store)

        out = _soil_moisture_preprocessor(ctx, "sm")

        self.assertIn("SWI", out)

        out_time = pd.to_datetime(out.time.values)

        expected_time = pd.date_range("2020-01-01", "2020-01-03", freq="D")

        self.assertTrue(out_time.equals(expected_time))
        self.assertEqual(len(out_time), 3)

    def test_land_cover_preprocessor(self):
        lc = make_lc_ds()
        ctx = DummyContext(store=None)

        out = _land_cover_preprocessor(ctx, lc)

        self.assertEqual(out.dtype, "uint8")
        self.assertEqual(out.shape, (2, 2))
        self.assertIn(1, out.values)
        self.assertEqual("lccs_class", out.name)
        self.assertEqual([[1, 1], [0, 1]], out.values.tolist())

    def test_era5_preprocessor(self):
        store = Mock()
        store.open_data.return_value = make_era5_ds()

        ctx = DummyContext(store)

        out = _era5_preprocessor(ctx, "era5")

        print(out.tp.values.tolist())

        self.assertEqual(out.pev.dtype, "int64")
        self.assertEqual(out.tp.dtype, "int64")
        self.assertEqual(out.pev.shape, (2, 2, 2))
        self.assertEqual(out.tp.shape, (2, 2, 2))
        self.assertEqual(
            [[[1000, -2000], [3000, -4000]], [[-2000, 3000], [-4000, -5000]]],
            out.pev.values.tolist(),
        )
        self.assertEqual(
            [[[10000, 20000], [30000, 40000]], [[20000, 30000], [40000, 50000]]],
            out.tp.values.tolist(),
        )

    def test_swicomp_nan_all_nan(self):
        data = np.array([np.nan, np.nan, np.nan])
        jd = np.array([1.0, 2.0, 3.0])

        out = _swicomp_nan(data, jd)

        self.assertTrue(np.all(np.isnan(out)))

    def test_swicomp_nan_single_value(self):
        data = np.array([np.nan, 10.0, np.nan])
        jd = np.array([1.0, 2.0, 3.0])

        out = _swicomp_nan(data, jd)

        self.assertTrue(np.isnan(out[0]))
        self.assertEqual(out[1], 10.0)
        self.assertTrue(np.isnan(out[0]))
        self.assertTrue(np.isnan(out[2]))

    def test_swicomp_nan_multiple_values(self):
        data = np.array([10.0, 20.0, 30.0, 40.0])
        jd = np.array([1.0, 2.0, 3.0, 4.0])

        out = _swicomp_nan(data, jd)

        # no NaNs
        self.assertFalse(np.any(np.isnan(out)))

        # first two values unchanged (loop starts at i=2)
        self.assertEqual(out[0], 10.0)
        self.assertEqual(out[1], 20.0)

        # filtered signal must be bounded
        self.assertTrue(out[2] <= data[2])
        self.assertTrue(out[3] <= data[3])

    def test_swicomp_nan_ctime_effect(self):
        data = np.array([10.0, 20.0, 30.0, 40.0])
        jd = np.array([1.0, 2.0, 3.0, 4.0])

        out_fast = _swicomp_nan(data, jd, ctime=0.5)
        out_slow = _swicomp_nan(data, jd, ctime=5.0)

        # Faster response should track input more closely
        self.assertTrue(out_fast[-1] > out_slow[-1])

    def test_resample_and_merge(self):
        swi = make_swi_ds()
        lc_mask = make_lc_mask()
        era5 = make_era5_daily_ds()

        out = _resample_and_merge(swi, lc_mask, era5)

        self.assertIsInstance(out, xr.Dataset)
        self.assertEqual(out.sizes["time"], 3)
        self.assertEqual(out.sizes["lat"], 2)
        self.assertEqual(out.sizes["lon"], 2)

        self.assertIn("SWI", out)
        self.assertIn("pev", out)
        self.assertIn("tp", out)

        # corresponds to lc_mask == 0
        masked_lat = 4
        masked_lon = 4

        self.assertTrue(np.isnan(out["SWI"].sel(lat=masked_lat, lon=masked_lon)).all())
        self.assertTrue(np.isnan(out["pev"].sel(lat=masked_lat, lon=masked_lon)).all())

        # where lc_mask == 1
        unmasked_lat = 4
        unmasked_lon = 5

        self.assertFalse(
            np.isnan(out["SWI"].sel(lat=unmasked_lat, lon=unmasked_lon)).any()
        )
