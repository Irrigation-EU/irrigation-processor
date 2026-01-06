import unittest
from datetime import datetime
from unittest.mock import Mock

import xarray as xr

from irrigation_processor.core import XcubeDataStoreStorage
from irrigation_processor.core.pipeline import StepRegistry
from irrigation_processor.utils import (
    convert_m_to_mm,
    get_existing_data,
    inject_dynamic_context_from_config,
    split_date_range,
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
        data = xr.DataArray(
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
        data = xr.DataArray(2.0, attrs={"units": "m"})

        converted = convert_m_to_mm(data)

        self.assertEqual(converted.values, 2000)
        self.assertEqual(
            converted.attrs["long_name"],
            "Potential evaporation (millimeters)",
        )

    def test_convert_without_updating_long_name(self):
        data = xr.DataArray(
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


class TestGetExistingData(unittest.TestCase):
    def setUp(self):
        self.store = Mock()
        self.data_id = "test-data"

    def test_returns_none_when_data_id_not_in_store(self):
        self.store.list_data_ids.return_value = []

        result = get_existing_data(
            store=self.store,
            data_id=self.data_id,
            load=False,
        )

        self.assertIsNone(result)
        self.store.open_data.assert_not_called()

    def test_returns_data_id_when_exists_and_not_loaded(self):
        self.store.list_data_ids.return_value = [self.data_id]

        result = get_existing_data(
            store=self.store,
            data_id=self.data_id,
            load=False,
        )

        self.assertEqual(result, self.data_id)
        self.store.open_data.assert_not_called()

    def test_returns_dataset_when_exists_and_loaded(self):
        ds = xr.Dataset()
        self.store.list_data_ids.return_value = [self.data_id]
        self.store.open_data.return_value = ds

        result = get_existing_data(
            store=self.store,
            data_id=self.data_id,
            load=True,
        )

        self.assertIs(result, ds)
        self.store.open_data.assert_called_once_with(self.data_id)
