import inspect
import unittest

import xarray as xr
from pydantic import BaseModel

from irrigation_processor.constants import (
    LC_DATA_ID,
    INPUT_FOR_CALIBRATION_ID,
)
from irrigation_processor.core.pipeline import FromStep
from irrigation_processor.steps import registry


class TestPipelineDefinition(unittest.TestCase):
    """
    We explicitly DO NOT execute the steps.
    These tests verify declarative correctness only.
    """

    def test_all_steps_registered(self):
        steps = registry.all(include_disabled=True)
        names = {step.name for step in steps}

        self.assertEqual(
            names,
            {
                "dataloader",
                "preprocessing",
                "calibration",
                "simulation",
                "postprocessing",
            },
        )

    def test_dataloader_step_metadata(self):
        meta = registry.get("dataloader")

        self.assertEqual(meta.name, "dataloader")
        self.assertEqual(
            meta.outputs,
            ("sm_data_id", LC_DATA_ID, "era5_data_id"),
        )
        self.assertEqual(meta.inputs, ())
        self.assertIsNone(meta.context_cls)

    def test_preprocessing_step_metadata(self):
        meta = registry.get("preprocessing")

        self.assertEqual(meta.outputs, (INPUT_FOR_CALIBRATION_ID,))
        self.assertEqual(
            meta.inputs,
            (
                FromStep("dataloader", "sm_data_id"),
                FromStep("dataloader", LC_DATA_ID),
                FromStep("dataloader", "era5_data_id"),
            ),
        )

    def test_calibration_step_metadata(self):
        meta = registry.get("calibration")

        self.assertEqual(meta.outputs, ("calibrated_data_id",))
        self.assertEqual(
            meta.inputs,
            (FromStep("preprocessing", INPUT_FOR_CALIBRATION_ID),),
        )

    def test_simulation_step_metadata(self):
        meta = registry.get("simulation")

        self.assertEqual(
            meta.outputs,
            (
                "iwu_spatial_estimates",
                "iwu_temporal_estimates",
            ),
        )
        self.assertEqual(
            meta.inputs,
            (
                FromStep("preprocessing", INPUT_FOR_CALIBRATION_ID),
                FromStep("calibration", "calibrated_data_id"),
            ),
        )

    def test_postprocessing_step_metadata(self):
        meta = registry.get("postprocessing")

        self.assertEqual(meta.outputs, ())
        self.assertEqual(
            meta.inputs,
            (
                FromStep("simulation", "iwu_spatial_estimates"),
                FromStep("simulation", "iwu_temporal_estimates"),
            ),
        )

    def test_step_function_signatures(self):
        """
        Ensures step functions are compatible with LocalService expectations:
        - context first
        - dask_client as last arg
        """
        dataloader = registry.get("dataloader").func
        preprocessing = registry.get("preprocessing").func
        calibration = registry.get("calibration").func
        simulation = registry.get("simulation").func
        postprocessing = registry.get("postprocessing").func

        sig = inspect.signature(dataloader)
        params = list(sig.parameters.values())

        self.assertGreater(len(params), 0)
        self.assertEqual(params[0].name, "context")
        self.assertEqual(params[0].annotation, BaseModel)

        sig = inspect.signature(preprocessing)
        params = list(sig.parameters.values())

        self.assertGreater(len(params), 1)
        self.assertIn("lc_cube", sig.parameters)
        self.assertEqual(sig.parameters["lc_cube"].annotation, xr.Dataset)

        for step in (calibration, simulation, postprocessing):
            sig = inspect.signature(step)
            params = list(sig.parameters.values())
            param_names = [p.name for p in params]

            self.assertIn("dask_client", param_names)

            self.assertEqual(param_names[-1], "dask_client")
            self.assertEqual(param_names[0], "context")
