import unittest
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import xarray as xr

from irrigation_processor.constants import (
    IWU_POSTPROCESSED_ESTIMATES_SPATIAL_ID,
    IWU_POSTPROCESSED_ESTIMATES_TEMPORAL_ID,
)
from irrigation_processor.ops import postprocessor
from irrigation_processor.ops.postprocessor import (
    _do_spatial_masking,
    _do_temporal_masking,
)


def make_iwu_ds():
    time = pd.date_range("2020-01-01", periods=3, freq="2W")
    return xr.Dataset(
        {
            "iwu_est": (
                ("time", "lat", "lon"),
                [[[10, 0], [5, 20]], [[30, 0], [10, 40]], [[0, 0], [0, 50]]],
            ),
        },
        coords={
            "time": time,
            "lat": [5, 4],
            "lon": [5, 6],
        },
    )


def make_mask_ds():
    return xr.Dataset(
        {"band_1": (("y", "x"), [[1.0, 0.0], [1.0, 1.0]])},
        coords={
            "y": [5, 4],
            "x": [5, 6],
        },
    )


class TestPostprocessor(unittest.TestCase):
    def test_postprocessor_cached(self):
        store = Mock()
        store.list_data_ids.return_value = [
            IWU_POSTPROCESSED_ESTIMATES_SPATIAL_ID,
            IWU_POSTPROCESSED_ESTIMATES_TEMPORAL_ID,
        ]

        ctx = Mock()
        ctx.store = store

        result = postprocessor(
            ctx,
            "spatial.zarr",
            "temporal.zarr",
            dask_client=None,
        )

        self.assertEqual(
            result,
            {
                "iwu_postprocessed_spatial_path": IWU_POSTPROCESSED_ESTIMATES_SPATIAL_ID,
                "iwu_postprocessed_temporal_path": IWU_POSTPROCESSED_ESTIMATES_TEMPORAL_ID,
            },
        )

    def test_do_temporal_masking(self):
        spatial = make_iwu_ds()
        temporal = make_iwu_ds()

        ctx = Mock()
        ctx.temporal_allowed_months = [1]  # January only

        spatial_out, temporal_out = _do_temporal_masking(ctx, spatial, temporal)
        self.assertFalse((temporal_out.isel(time=0)["iwu_est"] == 0).all())
        self.assertFalse((spatial_out.isel(time=0)["iwu_est"] == 0).all())
        self.assertFalse((spatial_out.isel(time=1)["iwu_est"] == 0).all())
        self.assertTrue((temporal_out.isel(time=2)["iwu_est"] == 0).all())

    @patch("irrigation_processor.ops.postprocessor._get_spatial_mask")
    def test_do_spatial_masking(
        self,
        mock_get_mask,
    ):
        spatial = make_iwu_ds()
        temporal = make_iwu_ds()

        ctx = Mock()
        ctx.spatial_mask_bbox = (5, 4, 6, 5)
        ctx.spatial_mask_threshold = 0.5

        mask_ds = make_mask_ds()
        mock_get_mask.return_value = mask_ds

        out_spatial, out_temporal = _do_spatial_masking(ctx, spatial, temporal)

        self.assertTrue(np.isnan(out_spatial["iwu_est"].sel(lat=5, lon=6)).all())
        self.assertTrue((out_spatial["iwu_est"].sel(lat=4, lon=5).values > 0).any())

    @patch("irrigation_processor.ops.postprocessor.get_existing_data")
    @patch("irrigation_processor.ops.postprocessor._do_spatial_masking")
    @patch("irrigation_processor.ops.postprocessor._do_temporal_masking")
    def test_postprocessor_flow(
        self,
        mock_temporal,
        mock_spatial,
        mock_get_existing_data,
    ):
        mock_get_existing_data.side_effect = [None, None]
        store = Mock()
        store.open_data.side_effect = [
            make_iwu_ds(),  # spatial
            make_iwu_ds(),  # temporal
        ]

        ctx = Mock()
        ctx.store = store
        ctx.bbox = [4, 5, 5, 6]
        ctx.time_range = ["2020-01-31", "2020-03-31"]

        mock_temporal.return_value = (
            make_iwu_ds(),
            make_iwu_ds(),
        )
        mock_spatial.return_value = (
            make_iwu_ds(),
            make_iwu_ds(),
        )

        result = postprocessor(
            ctx,
            "iwu_spatial.zarr",
            "iwu_temporal.zarr",
            dask_client=None,
        )

        # writes happened
        store.write_data.assert_any_call(
            unittest.mock.ANY,
            IWU_POSTPROCESSED_ESTIMATES_SPATIAL_ID,
        )
        store.write_data.assert_any_call(
            unittest.mock.ANY,
            IWU_POSTPROCESSED_ESTIMATES_TEMPORAL_ID,
        )

        self.assertEqual(
            result,
            {
                "iwu_postprocessed_spatial_path": IWU_POSTPROCESSED_ESTIMATES_SPATIAL_ID,
                "iwu_postprocessed_temporal_path": IWU_POSTPROCESSED_ESTIMATES_TEMPORAL_ID,
            },
        )
