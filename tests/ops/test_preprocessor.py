import unittest
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import xarray as xr

from irrigation_processor.constants import PROCESSED_CLMS_DATA_ID
from irrigation_processor.ops.preprocessor import (_era5_preprocessor,
                                                   _land_cover_preprocessor,
                                                   _resample_and_merge,
                                                   _soil_moisture_preprocessor,
                                                   _swicomp_nan,
                                                   irrigation_preprocessor)
from tests.helpers import (DummyContext, make_clms_ds, make_clms_ds_with_gap,
                           make_era5_ds, make_lc_ds, make_lc_mask)
from tests.helpers import make_preprocessed_ds as make_era5_daily_ds
from tests.helpers import make_preprocessed_ds as make_swi_ds


class TestPreprocessor(unittest.TestCase):
    @patch("irrigation_processor.ops.preprocessor.get_existing_data")
    def test_irrigation_preprocessor_cached(self, mock_get_existing_data):
        store = Mock()
        mock_get_existing_data.return_value = make_swi_ds()

        ctx = DummyContext()

        out = irrigation_preprocessor(ctx, store, "sm", "lc", "era5", "gleam")

        self.assertIsInstance(out, xr.Dataset)

    @patch("irrigation_processor.ops.preprocessor._resample_and_merge")
    @patch("irrigation_processor.ops.preprocessor._gleam_preprocessor")
    @patch("irrigation_processor.ops.preprocessor._era5_preprocessor")
    @patch("irrigation_processor.ops.preprocessor._land_cover_preprocessor")
    @patch("irrigation_processor.ops.preprocessor._soil_moisture_preprocessor")
    def test_irrigation_preprocessor_flow(
        self,
        mock_sm,
        mock_lc,
        mock_era5,
        mock_gleam,
        mock_merge,
    ):
        store = Mock()
        store.list_ids.return_value = []
        store.exists.return_value = False

        ctx = DummyContext()

        mock_sm.return_value = "SM"
        mock_lc.return_value = "LC"
        mock_era5.return_value = "ERA5"
        mock_merge.return_value = "MERGED"
        mock_gleam.return_value = None

        out = irrigation_preprocessor(ctx, store, "sm", "lc", "era5", "gleam")

        self.assertEqual(out, "MERGED")
        mock_merge.assert_called_once_with(
            "SM", "LC", "ERA5", None, chunk_sizes={"time": -1, "lat": 50, "lon": 50}
        )

    @patch("irrigation_processor.ops.preprocessor.validate_dataset")
    def test_soil_moisture_preprocessor_cached(self, mock_validate):
        store = Mock()
        store.list_data_ids.return_value = [PROCESSED_CLMS_DATA_ID]
        store.load.return_value = xr.Dataset()
        mock_validate.return_value = None

        ctx = DummyContext()

        out = _soil_moisture_preprocessor(ctx, store, "sm")

        self.assertTrue(out.equals(xr.Dataset()))

    def test_soil_moisture_preprocessor(self):
        store = Mock()
        store.exists.return_value = False
        store.list_ids.return_value = []
        store.load.return_value = make_clms_ds(
            time_range=pd.date_range("2024-01-01", periods=3, freq="D")
        )

        ctx = DummyContext()

        out = _soil_moisture_preprocessor(ctx, store, "sm")

        self.assertIn("SWI", out)
        self.assertEqual(out["SWI"].dims, ("time", "lat", "lon"))
        self.assertEqual(out["SWI"].sizes, {"time": 3, "lat": 2, "lon": 2})

    def test_soil_moisture_preprocessor_fills_missing_dates(self):
        store = Mock()
        store.exists.return_value = False
        store.list_ids.return_value = []
        store.load.return_value = make_clms_ds_with_gap()

        ctx = DummyContext()

        out = _soil_moisture_preprocessor(ctx, store, "sm")

        self.assertIn("SWI", out)

        out_time = pd.to_datetime(out.time.values)

        expected_time = pd.date_range("2024-01-01", "2024-01-03", freq="D")

        self.assertTrue(out_time.equals(expected_time))
        self.assertEqual(len(out_time), 3)

    def test_land_cover_preprocessor(self):
        store = Mock()
        store.load.return_value = make_lc_ds()
        ctx = DummyContext()
        ctx.preprocessing.lc_keep_classes = [10, 30]

        out = _land_cover_preprocessor(ctx, store, "lc")

        self.assertEqual(out.dtype, "bool")
        self.assertEqual(out.shape, (2, 2))
        self.assertIn(1, out.values)
        self.assertEqual("lccs_class", out.name)
        self.assertEqual([[1, 1], [0, 0]], out.values.tolist())

    def test_era5_preprocessor(self):
        store = Mock()
        store.load.return_value = make_era5_ds(
            time_range=pd.date_range("2024-01-01", periods=2, freq="D")
        )

        ctx = DummyContext()

        out = _era5_preprocessor(ctx, store, "era5")

        self.assertEqual(out.pev.dtype, "float32")
        self.assertEqual(out.tp.dtype, "float32")
        self.assertEqual(out.pev.shape, (2, 2, 2))
        self.assertEqual(out.tp.shape, (2, 2, 2))
        self.assertEqual(
            [
                [[1000.0, -2000.0], [3000.0, -4000.0]],
                [[-2000.0, 3000.0], [-4000.0, -5000.0]],
            ],
            out.pev.values.tolist(),
        )
        self.assertEqual(
            [[10000.0, 20000.0], [30000.0, 40000.0]],
            out.tp.isel(time=0).values.tolist(),
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
        time = pd.date_range("2024-01-01", periods=3, freq="D")
        swi = make_swi_ds(time_range=time)
        lc_mask = make_lc_mask()
        era5 = make_era5_daily_ds(time_range=time)

        out = _resample_and_merge(swi, lc_mask, era5)

        self.assertIsInstance(out, xr.Dataset)
        self.assertEqual(out.sizes["time"], 3)
        self.assertEqual(out.sizes["lat"], 2)
        self.assertEqual(out.sizes["lon"], 2)

        self.assertIn("SWI", out)
        self.assertIn("pev", out)
        self.assertIn("tp", out)

        # corresponds to lc_mask == 1
        unmasked_lat = 44
        unmasked_lon = -5

        self.assertFalse(
            np.isnan(out["SWI"].sel(lat=unmasked_lat, lon=unmasked_lon)).all()
        )
        self.assertFalse(
            np.isnan(out["pev"].sel(lat=unmasked_lat, lon=unmasked_lon)).all()
        )

        # where lc_mask == 0
        masked_lat = 44
        masked_lon = -4

        self.assertTrue(np.isnan(out["SWI"].sel(lat=masked_lat, lon=masked_lon)).any())
