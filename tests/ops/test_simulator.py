import unittest
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import xarray as xr

from irrigation_processor.constants import (IWU_ESTIMATES_SPATIAL_ID,
                                            IWU_ESTIMATES_TEMPORAL_ID)
from irrigation_processor.ops import irrigation_simulator
from irrigation_processor.ops.simulator import _resample_sum, _ts_smet4irr
from tests.helpers import (DummyContext, make_calibration_ds, make_iwu_est_ds,
                           make_preprocessed_ds)


class TestSimulator(unittest.TestCase):
    def test_basic_simulation(self):
        sm = np.array([0.2, 0.25, 0.3, 0.35, 0.4])
        et = np.array([0.005, 0.006, 0.007, 0.008, 0.009])
        a, b, z, RF = 10.0, 2.0, 100.0, 1.0

        result = _ts_smet4irr(sm, et, a, b, z, RF)

        # Output should be one element shorter than input
        self.assertEqual(len(result), len(sm) - 1)

        # All values should be non-negative
        self.assertTrue(np.all(result >= 0))

    def test_small_changes_filtered(self):
        sm = np.array([0.3, 0.3005, 0.301, 0.3005, 0.3])
        et = np.array([0.005, 0.005, 0.005, 0.005, 0.005])
        a, b, z, RF = 10.0, 2.0, 100.0, 1.0

        result = _ts_smet4irr(sm, et, a, b, z, RF)

        # Small changes should be filtered
        self.assertTrue(np.all(result == 0))

    def test_threshold_clipping(self):
        sm = np.array([0.1, 0.5, 0.9])
        et = np.array([0.01, 0.01, 0.01])
        a, b, z, RF = 50.0, 2.0, 200.0, 1.0
        thr = 30.0

        result = _ts_smet4irr(sm, et, a, b, z, RF, thr=thr)

        # All values should be <= threshold
        self.assertTrue(np.all(result <= thr))

    def test_values_below_1mm_filtered(self):
        sm = np.array([0.2, 0.205, 0.21, 0.215])
        et = np.array([0.001, 0.001, 0.001, 0.001])
        a, b, z, RF = 0.5, 1.5, 20.0, 0.8

        result = _ts_smet4irr(sm, et, a, b, z, RF)

        # Should be zeros or >= 1.0
        self.assertTrue(np.all((result == 0) | (result >= 1.0)))

    @patch("irrigation_processor.ops.simulator.chunk_dataset")
    def test_irrigation_simulator_cached(self, mock_chunk):
        store = Mock()
        store.exists.return_value = True
        store.list_ids.return_value = [
            IWU_ESTIMATES_SPATIAL_ID,
            IWU_ESTIMATES_TEMPORAL_ID,
        ]

        ctx = DummyContext()

        result = irrigation_simulator(
            ctx,
            store,
            preprocessed_ds=make_preprocessed_ds(),
            calibrated_path="cal.zarr",
        )

        self.assertEqual(
            result,
            {
                "iwu_spatial_estimates": IWU_ESTIMATES_SPATIAL_ID,
                "iwu_temporal_estimates": IWU_ESTIMATES_TEMPORAL_ID,
            },
        )

    def test_resample_sum_weekly(self):
        time = pd.date_range("2024-01-01", periods=14, freq="D")
        da = xr.DataArray(
            np.ones((14, 2, 2)),
            dims=("time", "lat", "lon"),
            coords={
                "time": time,
                "lat": [44.0, 43.0],
                "lon": [-5.0, -4.0],
                "spatial_ref": 0,
            },
        )

        out = _resample_sum(da, step=7)

        self.assertEqual(out.sizes["time"], 2)
        self.assertTrue(np.all(out.values == 7.0))

        out2 = _resample_sum(out, step=2)

        self.assertEqual(out2.sizes["time"], 1)
        self.assertTrue(np.all(out2.values == 14.0))

    def test_irrigation_simulator_flow(
        self,
    ):
        store = Mock()
        store.exists.side_effect = [False, False, True, True]  # Check inputs
        store.list_ids.return_value = []
        store.load.side_effect = [
            make_calibration_ds(),  # open calibrated_path
            make_iwu_est_ds(),  # open temporal result
        ]

        ctx = DummyContext()

        pre = make_preprocessed_ds(
            time_range=pd.date_range("2024-01-01", periods=61, freq="D")
        )

        result = irrigation_simulator(
            ctx,
            store,
            preprocessed_ds=pre,
            calibrated_path="cal.zarr",
        )

        self.assertEqual(
            result,
            {
                "iwu_spatial_estimates": IWU_ESTIMATES_SPATIAL_ID,
                "iwu_temporal_estimates": IWU_ESTIMATES_TEMPORAL_ID,
            },
        )

        temporal_calls = [
            c for c in store.save.call_args_list if c[0][0] == IWU_ESTIMATES_TEMPORAL_ID
        ]
        self.assertEqual(len(temporal_calls), 1)

        temporal_ds = temporal_calls[0][0][1]
        self.assertIn("iwu_est", temporal_ds)
        self.assertEqual(temporal_ds["iwu_est"].dims, ("time", "lat", "lon"))

        spatial_calls = [
            c for c in store.save.call_args_list if c[0][0] == IWU_ESTIMATES_SPATIAL_ID
        ]
        self.assertEqual(len(spatial_calls), 1)
