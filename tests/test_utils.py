import unittest
from datetime import datetime

from unittest.mock import Mock

from xarray import DataArray

from irrigation_processor.core import XcubeDataStoreStorage
from irrigation_processor.core.pipeline import StepRegistry
from irrigation_processor.utils import (
    split_date_range,
    convert_m_to_mm,
    inject_dynamic_context_from_config,
)


class TestSplitDateRange(unittest.TestCase):
    def test_split_date_range_with_datetime(self):
        start = datetime(2023, 1, 1)
        end = datetime(2023, 1, 31)

        result = split_date_range(start, end, num_days=10)

        self.assertEqual(
            result,
            [
                ("2023-01-01", "2023-01-10"),
                ("2023-01-11", "2023-01-20"),
                ("2023-01-21", "2023-01-30"),
                ("2023-01-31", "2023-01-31"),
            ],
        )

    def test_split_date_range_with_string_inputs(self):
        result = split_date_range("2023-01-01", "2023-01-05", num_days=3)

        self.assertEqual(
            result,
            [
                ("2023-01-01", "2023-01-03"),
                ("2023-01-04", "2023-01-05"),
            ],
        )

    def test_split_date_range_single_day(self):
        result = split_date_range("2023-01-01", "2023-01-01")

        self.assertEqual(result, [("2023-01-01", "2023-01-01")])


class TestConvertMToMM(unittest.TestCase):
    def test_convert_with_existing_long_name(self):
        data = DataArray(
            1.5,
            attrs={
                "units": "m",
                "GRIB_units": "m",
                "long_name": "Potential evaporation",
            },
        )

        converted = convert_m_to_mm(data)

        self.assertEqual(converted.values, 1500)
        self.assertEqual(converted.attrs["units"], "mm")
        self.assertEqual(converted.attrs["GRIB_units"], "mm")
        self.assertEqual(
            converted.attrs["long_name"],
            "Potential evaporation (millimeters)",
        )

    def test_convert_without_long_name(self):
        data = DataArray(2.0, attrs={"units": "m"})

        converted = convert_m_to_mm(data)

        self.assertEqual(converted.values, 2000)
        self.assertEqual(
            converted.attrs["long_name"],
            "Potential evaporation (millimeters)",
        )

    def test_convert_without_updating_long_name(self):
        data = DataArray(
            1.0,
            attrs={"long_name": "Evaporation", "units": "m"},
        )

        converted = convert_m_to_mm(data, update_long_name=False)

        self.assertEqual(converted.attrs["long_name"], "Evaporation")


class DummyStepMeta:
    def __init__(self, name):
        self.name = name
        self.context_cls = None


class TestInjectDynamicContextFromConfig(unittest.TestCase):
    def setUp(self):
        self.step1 = DummyStepMeta("step1")
        self.step2 = DummyStepMeta("step2")

        self.registry = Mock(spec=StepRegistry)
        self.registry.all.return_value = [self.step1, self.step2]

        self.storage = Mock(spec=XcubeDataStoreStorage)
        self.storage.store = "STORE_OBJECT"

    def test_successful_context_injection(self):
        config = {
            "base": {"a": 1},
            "step1": {"b": 2},
            "step2": {"c": 3},
        }

        inject_dynamic_context_from_config(
            config=config,
            registry=self.registry,
            storage=self.storage,
        )

        ctx1 = self.step1.context_cls()
        ctx2 = self.step2.context_cls()

        self.assertEqual(ctx1.a, 1)
        self.assertEqual(ctx1.b, 2)
        self.assertEqual(ctx1.store, "STORE_OBJECT")

        self.assertEqual(ctx2.a, 1)
        self.assertEqual(ctx2.c, 3)
        self.assertEqual(ctx2.store, "STORE_OBJECT")

    def test_unknown_steps_raise_value_error(self):
        config = {
            "base": {},
            "step1": {},
            "unknown": {},
        }

        with self.assertRaises(ValueError) as ctx:
            inject_dynamic_context_from_config(
                config=config,
                registry=self.registry,
                storage=self.storage,
            )

        self.assertIn("Unknown steps in config", str(ctx.exception))

    def test_base_config_not_dict_raises_type_error(self):
        config = {
            "base": "not-a-dict",
            "step1": {},
        }

        with self.assertRaises(TypeError) as ctx:
            inject_dynamic_context_from_config(
                config=config,
                registry=self.registry,
                storage=self.storage,
            )

        self.assertIn("'base' config must be a dict", str(ctx.exception))

    def test_step_config_not_dict_raises_type_error(self):
        config = {
            "base": {},
            "step1": "invalid",
        }

        with self.assertRaises(TypeError) as ctx:
            inject_dynamic_context_from_config(
                config=config,
                registry=self.registry,
                storage=self.storage,
            )

        self.assertIn("Config for step 'step1' must be a dict", str(ctx.exception))
