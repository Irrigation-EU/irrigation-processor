import unittest
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import xarray as xr

from irrigation_processor.constants import (
    IWU_POSTPROCESSED_ESTIMATES_SPATIAL_ID,
    IWU_POSTPROCESSED_ESTIMATES_TEMPORAL_ID)
from irrigation_processor.ops import postprocessor
from irrigation_processor.ops.postprocessor import (_do_spatial_masking,
                                                    _do_temporal_masking)
from tests.helpers import DummyContext, make_iwu_ds, make_mask_ds


class TestPostprocessor(unittest.TestCase):
    def test_postprocessor_cached(self):
        store = Mock()
        store.exists.side_effect = [True, True] # Check both ids
        store.list_ids.return_value = [
            IWU_POSTPROCESSED_ESTIMATES_SPATIAL_ID,
            IWU_POSTPROCESSED_ESTIMATES_TEMPORAL_ID,
        ]

        ctx = DummyContext()

        result = postprocessor(
            ctx,
            store,
            "spatial.zarr",
            "temporal.zarr",
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

        ctx = DummyContext()
        ctx.postprocessing.temporal_allowed_months = [1]  # January only

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

        ctx = DummyContext()

        mask_ds = make_mask_ds()
        mock_get_mask.return_value = mask_ds

        out_spatial, out_temporal = _do_spatial_masking(ctx, spatial, temporal)
        self.assertTrue(np.isnan(out_spatial["iwu_est"].sel(lat=44, lon=-4)).all())
        self.assertTrue((out_spatial["iwu_est"].sel(lat=43, lon=-5).values > 0).any())

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
        store.load.side_effect = [
            make_iwu_ds(),  # spatial
            make_iwu_ds(),  # temporal
        ]

        ctx = DummyContext()
        ctx.base.bbox = [-5, 43, -4, 44]
        ctx.base.time_range = ["2024-01-31", "2024-03-31"]

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
            store,
            "iwu_spatial.zarr",
            "iwu_temporal.zarr",
        )

        # writes happened
        store.save.assert_any_call(
            IWU_POSTPROCESSED_ESTIMATES_SPATIAL_ID,
            unittest.mock.ANY,
        )
        store.save.assert_any_call(
            IWU_POSTPROCESSED_ESTIMATES_TEMPORAL_ID,
            unittest.mock.ANY,
        )

        self.assertEqual(
            result,
            {
                "iwu_postprocessed_spatial_path": IWU_POSTPROCESSED_ESTIMATES_SPATIAL_ID,
                "iwu_postprocessed_temporal_path": IWU_POSTPROCESSED_ESTIMATES_TEMPORAL_ID,
            },
        )
