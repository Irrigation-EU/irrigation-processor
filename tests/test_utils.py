import unittest
from datetime import datetime
from unittest.mock import Mock

import numpy as np
import pandas as pd
import xarray as xr
from pydantic import BaseModel

from irrigation_processor.utils import (convert_m_to_mm, get_existing_data,
                                        split_date_range, validate_dataset)


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


class TestGetExistingData(unittest.TestCase):
    def setUp(self):
        self.store = Mock()
        self.data_id = "test-data"

    def test_returns_none_when_data_id_not_in_store(self):
        self.store.exists.return_value = False

        result = get_existing_data(
            storage=self.store,
            data_id=self.data_id,
            load=False,
        )

        self.assertIsNone(result)
        self.store.open_data.assert_not_called()

    def test_returns_data_id_when_exists_and_not_loaded(self):
        self.store.exists.return_value = True

        result = get_existing_data(
            storage=self.store,
            data_id=self.data_id,
            load=False,
        )

        self.assertEqual(result, self.data_id)
        self.store.open_data.assert_not_called()

    def test_returns_dataset_when_exists_and_loaded(self):
        ds = xr.Dataset()
        self.store.exists.return_value = True
        self.store.load.return_value = ds

        result = get_existing_data(
            storage=self.store,
            data_id=self.data_id,
            load=True,
        )

        self.assertIs(result, ds)
        self.store.load.assert_called_once_with(self.data_id)


class TestValidateDataset(unittest.TestCase):
    def setUp(self):
        lat = np.arange(44, 39.9, -0.5)  # descending
        lon = np.arange(-5, 3.1, 0.5)
        time = pd.date_range("2024-01-01", "2024-01-10")

        data = np.random.rand(len(time), len(lat), len(lon))

        self.dataset = xr.Dataset(
            {"var": (("time", "lat", "lon"), data)},
            coords={
                "time": time,
                "lat": lat,
                "lon": lon,
            },
        )

        class BaseConfig(BaseModel):
            bbox: list[float] = [-5, 40, 3, 44]
            time_range: list[str] = ["2024-01-02", "2024-01-08"]
            use_gleam: bool = False

        class Context(BaseModel):
            base: BaseConfig = BaseConfig()

        self.context = Context()

    def test_valid_dataset_passes(self):
        validate_dataset(self.context, self.dataset)

    def test_missing_coordinates_raises(self):
        ds = self.dataset.drop_dims("lat")

        with self.assertRaises(ValueError):
            validate_dataset(self.context, ds)

    def test_latitude_not_descending_raises(self):
        ds = self.dataset.sortby("lat")  # ascending

        with self.assertRaises(ValueError):
            validate_dataset(self.context, ds)

    def test_latitude_coverage_failure(self):
        ds = self.dataset.sel(lat=slice(43, 41))

        with self.assertRaises(ValueError):
            validate_dataset(self.context, ds)

    def test_longitude_coverage_failure(self):
        ds = self.dataset.sel(lon=slice(-4, 2))

        with self.assertRaises(ValueError):
            validate_dataset(self.context, ds)

    def test_temporal_coverage_failure(self):
        ds = self.dataset.sel(time=slice("2024-01-03", "2024-01-06"))

        with self.assertRaises(ValueError):
            validate_dataset(self.context, ds)

    def test_spatial_subset_empty(self):
        bad_context = self.context
        bad_context.base.bbox = [100, 100, 110, 110]

        with self.assertRaises(ValueError):
            validate_dataset(bad_context, self.dataset)

    def test_temporal_subset_empty(self):
        bad_context = self.context
        bad_context.base.time_range = ["2030-01-01", "2030-01-10"]

        with self.assertRaises(ValueError):
            validate_dataset(bad_context, self.dataset)
