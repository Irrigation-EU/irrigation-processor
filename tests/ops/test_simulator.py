import unittest
from unittest.mock import Mock

import numpy as np
import pandas as pd
import xarray as xr

from irrigation_processor.constants import (
    IWU_ESTIMATES_SPATIAL_ID,
    IWU_ESTIMATES_TEMPORAL_ID,
)
from irrigation_processor.ops import irrigation_simulator
from irrigation_processor.ops.simulator import _resample_sum, _ts_smet4irr


def make_preprocessed_ds():
    time = pd.date_range("2020-01-01", periods=8, freq="D")  # enough for weekly
    return xr.Dataset(
        {
            "SWI": (("time", "lat", "lon"), np.ones((8, 2, 2)) * 0.3),
            "pev": (("time", "lat", "lon"), np.ones((8, 2, 2)) * 2.0),
            "tp": (("time", "lat", "lon"), np.ones((8, 2, 2)) * 1.0),
        },
        coords={
            "time": time,
            "lat": [4, 5],
            "lon": [5, 6],
            "spatial_ref": 0,
        },
    )


def make_calibration_ds():
    return xr.Dataset(
        {
            "calibration": (
                ("lat", "lon", "params"),
                np.ones((2, 2, 4)),
            )
        },
        coords={
            "lat": [4, 5],
            "lon": [5, 6],
            "params": ["a", "b", "z", "RF"],
        },
    )


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
        """Test threshold parameter clips maximum values"""
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

    def test_irrigation_simulator_cached(self):
        store = Mock()
        store.list_data_ids.return_value = [
            IWU_ESTIMATES_SPATIAL_ID,
            IWU_ESTIMATES_TEMPORAL_ID,
        ]

        ctx = Mock()
        ctx.store = store

        result = irrigation_simulator(
            ctx,
            preprocessed_ds=make_preprocessed_ds(),
            calibrated_path="cal.zarr",
            dask_client=None,
        )

        self.assertEqual(
            result,
            {
                "iwu_spatial_estimates": IWU_ESTIMATES_SPATIAL_ID,
                "iwu_temporal_estimates": IWU_ESTIMATES_TEMPORAL_ID,
            },
        )

    def test_resample_sum_weekly(self):
        time = pd.date_range("2020-01-01", periods=14, freq="D")
        da = xr.DataArray(
            np.ones((14, 2, 2)),
            dims=("time", "lat", "lon"),
            coords={
                "time": time,
                "lat": [4, 5],
                "lon": [5, 6],
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
        store.list_data_ids.return_value = []
        store.open_data.side_effect = [
            make_calibration_ds(),  # open calibrated_path
            xr.Dataset(
                {"iwu_est": (("time", "lat", "lon"), np.ones((2, 2, 2)))}
            ),  # open temporal result
        ]

        ctx = Mock()
        ctx.store = store

        pre = make_preprocessed_ds()

        result = irrigation_simulator(
            ctx,
            preprocessed_ds=pre,
            calibrated_path="cal.zarr",
            dask_client=None,
        )

        self.assertEqual(
            result,
            {
                "iwu_spatial_estimates": IWU_ESTIMATES_SPATIAL_ID,
                "iwu_temporal_estimates": IWU_ESTIMATES_TEMPORAL_ID,
            },
        )

        temporal_calls = [
            c
            for c in store.write_data.call_args_list
            if c[0][1] == IWU_ESTIMATES_TEMPORAL_ID
        ]
        self.assertEqual(len(temporal_calls), 1)

        temporal_ds = temporal_calls[0][0][0]
        self.assertIn("iwu_est", temporal_ds)
        self.assertEqual(temporal_ds["iwu_est"].dims, ("time", "lat", "lon"))

        spatial_calls = [
            c
            for c in store.write_data.call_args_list
            if c[0][1] == IWU_ESTIMATES_SPATIAL_ID
        ]
        self.assertEqual(len(spatial_calls), 1)
