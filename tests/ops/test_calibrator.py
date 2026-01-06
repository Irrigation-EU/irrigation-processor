import unittest
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import xarray as xr

from irrigation_processor.constants import CALIBRATED_ID
from irrigation_processor.ops.calibrator import (
    calib_sm_inversion,
    calib_wrapper,
    cost_fun,
    sm_inversion,
    soil_moisture_inversion_calibration,
)


class DummyContext:
    def __init__(self, store):
        self.store = store


def make_calibrated_ds():
    return xr.Dataset(
        {
            "calibration": (
                ("lat", "lon", "params"),
                np.array(
                    [
                        [[1.0, 2.0, 3.0, 4.0]],
                        [[np.nan, np.nan, np.nan, np.nan]],
                    ]
                ),
            )
        },
        coords={
            "lat": [4, 5],
            "lon": [5],
            "params": ["a", "b", "z", "RF"],
        },
    )


class TestCalibrator(unittest.TestCase):
    def test_sm_inversion_basic_behavior(self):
        sm = np.array([0.2, 0.3, 0.4, 0.5])
        et = np.array([1.0, 1.0, 1.0, 1.0])

        out = sm_inversion(sm, et, a=1.0, b=1.0, z=10.0, RF=0.5)

        self.assertEqual(len(out), len(sm) - 1)
        self.assertTrue(np.all(out >= 0.0))

    def test_sm_inversion_small_changes_zeroed(self):
        sm = np.array([0.2, 0.2005, 0.2006])
        et = np.array([1.0, 1.0, 1.0])

        out = sm_inversion(sm, et, a=1.0, b=1.0, z=10.0, RF=0.5)

        self.assertTrue(np.all(out == 0.0))

    def test_sm_inversion_threshold_clipping(self):
        sm = np.array([0.1, 1.0])
        et = np.array([10.0, 10.0])

        out = sm_inversion(sm, et, a=100, b=2, z=100, RF=10, thr=5.0)

        self.assertTrue(np.all(out <= 5.0))

    def test_values_below_1mm_filtered(self):
        sm = np.array([0.2, 0.201, 0.202])
        et = np.array([0.001, 0.001, 0.001])
        a, b, z, RF = 0.1, 2.0, 10.0, 1.0

        result = sm_inversion(sm, et, a, b, z, RF)

        self.assertTrue(np.all((result == 0) | (result >= 1.0)))

    def test_cost_fun(self):
        sm = np.array([0.1, 0.2, 0.3, 0.4])
        et = np.ones_like(sm)
        p_obs = np.array([0, 0, 0.3, 0.4])

        cost = cost_fun(np.array([1, 1, 10, 0.5]), sm, p_obs, et, NN=1)
        self.assertAlmostEqual(np.round(cost, 2), 1.28)

    def test_cost_fun_handles_nans(self):
        sm = np.array([0.1, 0.2, 0.3, 0.4])
        et = np.ones_like(sm)
        p_obs = np.array([np.nan, 1.0, np.nan, 2.0])

        cost = cost_fun(np.array([1, 1, 10, 0.5]), sm, p_obs, et, NN=1)
        self.assertTrue(np.isfinite(cost))

    @patch("irrigation_processor.ops.calibrator.minimize")
    def test_calib_sm_inversion_calls_minimize(self, mock_minimize):
        mock_result = Mock()
        mock_result.x = np.array([1.0, 2.0, 3.0, 4.0])
        mock_minimize.return_value = mock_result

        sm = np.array([0.1, 0.2, 0.3])
        et = np.ones_like(sm)
        p_obs = np.ones(len(sm) - 1)

        a, b, z, RF = calib_sm_inversion(sm, p_obs, et, NN=1)

        self.assertEqual((a, b, z, RF), (1.0, 2.0, 3.0, 4.0))
        mock_minimize.assert_called_once()

    def test_calibration_returns_four_parameters(self):
        sm = np.linspace(0.2, 0.5, 100)
        p_obs = np.random.uniform(0, 10, 100)
        et = np.ones(100) * 0.005
        NN = 4

        a, b, z, RF = calib_sm_inversion(sm, p_obs, et, NN)

        self.assertIsInstance(a, (float, np.floating))
        self.assertIsInstance(b, (float, np.floating))
        self.assertIsInstance(z, (float, np.floating))
        self.assertIsInstance(RF, (float, np.floating))

    def test_calibration_respects_bounds(self):
        sm = np.linspace(0.2, 0.5, 100)
        p_obs = np.random.uniform(0, 10, 100)
        et = np.ones(100) * 0.005
        NN = 4

        bounds = ((0, 200), (0.01, 50), (1, 800), (0.1, 1.4))

        a, b, z, RF = calib_sm_inversion(sm, p_obs, et, NN, bounds=bounds)

        self.assertTrue(0 <= a <= 200)
        self.assertTrue(0.01 <= b <= 50)
        self.assertTrue(1 <= z <= 800)
        self.assertTrue(0.1 <= RF <= 1.4)

    def test_calib_wrapper_all_nan_returns_nan_params(self):
        sm = np.array([np.nan, np.nan])
        et = np.array([1.0, 1.0])
        p_obs = np.array([1.0, 1.0])

        out = calib_wrapper(sm, p_obs, et, NN=1)

        self.assertTrue(np.all(np.isnan(out)))

    @patch("irrigation_processor.ops.calibrator.calib_sm_inversion")
    def test_calib_wrapper_delegates(self, mock_calib):
        mock_calib.return_value = (1.0, 2.0, 3.0, 4.0)

        sm = np.array([0.1, 0.2, 0.3])
        et = np.ones_like(sm)
        p_obs = np.ones(len(sm) - 1)

        out = calib_wrapper(sm, p_obs, et, NN=1)
        self.assertTrue(np.allclose(out, [1, 2, 3, 4]))

    def test_calibration_cached(self):
        store = Mock()
        store.list_data_ids.return_value = [CALIBRATED_ID]
        store.open_data.return_value = xr.Dataset(
            {"calibration": (("lat", "lon", "params"), np.zeros((1, 1, 4)))}
        )

        ctx = DummyContext(store)
        ctx.check_calibration = False

        result = soil_moisture_inversion_calibration(
            ctx, xr.Dataset(), dask_client=None
        )

        self.assertEqual(result["calibrated_data_id"], CALIBRATED_ID)

    def test_calibration_flow(
        self,
    ):
        store = Mock()
        store.list_data_ids.side_effect = [
            [],  # no calibrated data
            ["calibrated_0.zarr"],
            ["calibrated_0.zarr"],
        ]
        store.has_data.return_value = False
        store.open_data.return_value = xr.Dataset(
            {"calibration": (("lat", "lon", "params"), np.zeros((1, 1, 4)))}
        )
        ctx = DummyContext(store)

        pre = xr.Dataset(
            {
                "SWI": (("time", "lat", "lon"), np.zeros((2, 1, 1))),
                "tp": (("time", "lat", "lon"), np.zeros((2, 1, 1))),
                "pev": (("time", "lat", "lon"), np.zeros((2, 1, 1))),
            },
            coords={"time": pd.to_datetime(("2020-01-01", "2020-01-02"))},
        )

        ctx.allowed_months = [1]
        ctx.rainfall_threshold = 0.1
        ctx.check_calibration = False

        result = soil_moisture_inversion_calibration(ctx, pre, dask_client=None)

        self.assertEqual(result["calibrated_data_id"], CALIBRATED_ID)

        written_ds = store.write_data.call_args_list[-1][0][0]
        self.assertEqual(written_ds.sizes["params"], 4)

    @patch("irrigation_processor.ops.calibrator.LOG")
    def test_cached_calibration_with_check(
        self,
        mock_log,
    ):
        store = Mock()
        store.list_data_ids.return_value = [CALIBRATED_ID]
        store.open_data.return_value = make_calibrated_ds()

        ctx = DummyContext(store)
        ctx.check_calibration = True

        result = soil_moisture_inversion_calibration(
            ctx,
            preprocessed_data=Mock(),
            dask_client=None,
        )

        self.assertEqual(
            result,
            {"calibrated_data_id": CALIBRATED_ID},
        )

        store.open_data.assert_called_once_with("calibrated.zarr")

        self.assertTrue(
            any(
                "Number of unique parameter sets" in str(c)
                for c in mock_log.info.call_args_list
            )
        )
        self.assertTrue(
            any(
                "Unique parameter sets without NaNs" in str(c)
                for c in mock_log.info.call_args_list
            )
        )
