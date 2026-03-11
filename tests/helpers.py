from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from irrigation_processor.config import (AppConfig, BaseConfig,
                                         CalibrationConfig, DaskConfig,
                                         DataloaderConfig,
                                         PostprocessingConfig,
                                         PreprocessingConfig, SimulationConfig,
                                         StorageConfig)

DEFAULT_BBOX = [-5.0, 43.0, -4.0, 44.0]
DEFAULT_TIME_RANGE = ["2024-01-01", "2024-01-02"]
DEFAULT_LAT = [44.0, 43.0]
DEFAULT_LON = [-5.0, -4.0]


class DummyStepMeta:
    def __init__(self, name, func=None, depends_on=None, inputs=None, outputs=None):
        self.name = name
        self.func = func
        self.depends_on = depends_on or []
        self.inputs = inputs or []
        self.outputs = outputs or []


class DummyContext:
    def __init__(self, store=None):
        self.store = store or Mock()

        self._config = make_test_config()

    def __getattr__(self, item):
        try:
            return getattr(self._config, item)
        except AttributeError:
            raise AttributeError(f"{type(self).__name__} has no attribute '{item}'")


def make_test_config() -> AppConfig:
    base_config = AppConfig(
        base=BaseConfig(
            bbox=DEFAULT_BBOX,
            time_range=DEFAULT_TIME_RANGE,
            use_gleam=False,
        ),
        dataloader=DataloaderConfig(
            lc_data_id="landcover",
            lc_time_range=["2024-01-01", "2024-12-31"],
            cds_data_id="era5",
            cds_spatial_res=0.1,
            cds_variable_names=["pev", "tp"],
            cds_optimize_writing=False,
            cds_intermediate_chunks={"time": 10, "lat": 178, "lon": 306},
            cds_final_chunks={"time": -1, "lat": 50, "lon": 50},
            cds_zappend_spatial_chunks=[15, 15],
            clms_zappend_spatial_chunks=[150, 150],
        ),
        preprocessing=PreprocessingConfig(
            lc_keep_classes=[10, 30],
            swi_chunks={"time": -1, "lat": 128, "lon": 128},
            merged_chunks={"time": -1, "lat": 50, "lon": 50},
        ),
        calibration=CalibrationConfig(
            allowed_months=list(range(1, 13)),
            rainfall_threshold=0.1,
            check_calibration=False,
            calibration_chunks={"params": 4, "lat": 50, "lon": 50},
        ),
        simulation=SimulationConfig(
            spatial_chunks={"time": 1, "lat": 50, "lon": 50},
        ),
        postprocessing=PostprocessingConfig(
            temporal_allowed_months=[1],
            spatial_mask_zip_url="http://example.com/mask.zip",
            spatial_mask_filename="mask.tif",
            spatial_mask_bbox=DEFAULT_BBOX,
            spatial_mask_threshold=5,
        ),
        storage=StorageConfig(
            store_id="store",
            store_kwargs={
                "root": "output_dir",
                "max_depth": 5,
                "storage_options": {
                    "anon": True,
                    "key": "key",
                    "secret": "secret",
                    "client_kwargs": {},
                },
            },
        ),
        dask=DaskConfig(
            dask_kwargs={
                "n_workers": 1,
                "threads_per_worker": 1,
                "memory_limit": "1GB",
            },
        ),
    )

    return AppConfig.model_validate(base_config)


@pytest.fixture
def dummy_context():
    return DummyContext


def make_preprocessed_ds(
    time_range=None,
    lat_range=None,
    lon_range=None,
    swi_data=None,
    tp_data=None,
    pev_data=None,
) -> xr.Dataset:
    if time_range is None:
        time = pd.date_range(DEFAULT_TIME_RANGE[0], periods=2, freq="D")
    else:
        time = pd.to_datetime(time_range)

    lat = np.array(lat_range) if lat_range is not None else np.array(DEFAULT_LAT)
    lon = np.array(lon_range) if lon_range is not None else np.array(DEFAULT_LON)

    shape = (len(time), len(lat), len(lon))

    if swi_data is None:
        swi_data = np.zeros(shape, dtype=np.float32) + 0.3
    if tp_data is None:
        tp_data = np.zeros(shape, dtype=np.float32) + 1.0
    if pev_data is None:
        pev_data = np.zeros(shape, dtype=np.float32) + 2.0

    return xr.Dataset(
        {
            "SWI": (("time", "lat", "lon"), swi_data),
            "tp": (("time", "lat", "lon"), tp_data),
            "pev": (("time", "lat", "lon"), pev_data),
        },
        coords={
            "time": time,
            "lat": lat,
            "lon": lon,
            "spatial_ref": 0,
        },
    )


def make_iwu_ds(time_range=None):
    if time_range is None:
        time = pd.date_range(DEFAULT_TIME_RANGE[0], periods=3, freq="2W")
    else:
        time = pd.to_datetime(time_range)

    shape = (len(time), 2, 2)
    return xr.Dataset(
        {
            "iwu_est": (
                ("time", "lat", "lon"),
                np.ones(shape, dtype=np.float32),
            ),
        },
        coords={
            "time": time,
            "lat": DEFAULT_LAT,
            "lon": DEFAULT_LON,
            "spatial_ref": 0,
        },
    )


def make_era5_ds(time_range=None):
    if time_range is None:
        time = pd.date_range(DEFAULT_TIME_RANGE[0], periods=2, freq="D")
    else:
        time = pd.to_datetime(time_range)

    pev = np.zeros((len(time), 2, 2), dtype=np.float32)
    tp = np.zeros((len(time), 2, 2), dtype=np.float32)

    if len(time) >= 1:
        pev[0] = [[-1, 2], [-3, 4]]
        tp[0] = [[10, 20], [30, 40]]
    if len(time) >= 2:
        pev[1] = [[2, -3], [4, 5]]
        tp[1] = [[20, 30], [40, 50]]

    return xr.Dataset(
        {
            "pev": (("time", "lat", "lon"), pev),
            "tp": (("time", "lat", "lon"), tp),
            "expver": ("time", [1] * len(time)),
            "number": ("time", [0] * len(time)),
        },
        coords={
            "time": time,
            "lat": DEFAULT_LAT,
            "lon": DEFAULT_LON,
            "spatial_ref": 0,
        },
    )


def make_clms_ds(time_range=None):
    if time_range is None:
        time = pd.date_range(DEFAULT_TIME_RANGE[0], periods=2, freq="D")
    else:
        time = pd.to_datetime(time_range)

    ssm = np.zeros((len(time), 2, 2), dtype=np.float32) + 0.1

    return xr.Dataset(
        {
            "ssm": (("time", "lat", "lon"), ssm),
        },
        coords={
            "time": time,
            "lat": DEFAULT_LAT,
            "lon": DEFAULT_LON,
            "spatial_ref": 0,
        },
    )


def make_raw_clms_ds(time_range=None):
    if time_range is None:
        time = pd.date_range(DEFAULT_TIME_RANGE[0], periods=2, freq="1D")
    else:
        time = pd.to_datetime(time_range)

    ssm = np.zeros((len(time), 2, 2), dtype=np.float32) + 0.1
    ssm_noise = np.zeros((len(time), 2, 2), dtype=np.float32) + 1.0

    return xr.Dataset(
        {
            "ssm": (("time", "y", "x"), ssm),
            "ssm_noise": (("time", "y", "x"), ssm_noise),
        },
        coords={
            "time": time,
            "y": DEFAULT_LAT,
            "x": DEFAULT_LON,
            "spatial_ref": 0,
        },
    )


def make_mask_ds():
    return xr.Dataset(
        {"band_1": (("y", "x"), [[6, 3], [7, 9]])},
        coords={
            "y": DEFAULT_LAT,
            "x": DEFAULT_LON,
            "spatial_ref": 0,
        },
    )


def make_clms_ds_with_gap():
    time = pd.to_datetime([DEFAULT_TIME_RANGE[0], "2024-01-03"])
    ssm = np.zeros((len(time), 2, 2), dtype=np.float32) + 0.1
    return xr.Dataset(
        {
            "ssm": (("time", "lat", "lon"), ssm),
        },
        coords={
            "time": time,
            "lat": DEFAULT_LAT,
            "lon": DEFAULT_LON,
            "spatial_ref": 0,
        },
    )


def make_lc_ds():
    return xr.Dataset(
        {"lccs_class": (("lat", "lon"), [[10, 30], [40, 11]])},
        coords={
            "lat": DEFAULT_LAT,
            "lon": DEFAULT_LON,
        },
    )


def make_lc_mask():
    return xr.DataArray(
        [[[1, 0], [1, 1]]],
        dims=("time", "lat", "lon"),
        coords={
            "time": [pd.to_datetime(DEFAULT_TIME_RANGE[0])],
            "lat": DEFAULT_LAT,
            "lon": DEFAULT_LON,
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
            "lat": DEFAULT_LAT,
            "lon": DEFAULT_LON,
            "params": ["a", "b", "z", "RF"],
        },
    )


def make_iwu_est_ds(time_range=None):
    if time_range is None:
        time = pd.date_range(DEFAULT_TIME_RANGE[0], periods=10, freq="2W")
    else:
        time = pd.to_datetime(time_range)

    return xr.Dataset(
        {"iwu_est": (("time", "lat", "lon"), np.ones((len(time), 2, 2)))},
        coords={
            "time": time,
            "lat": DEFAULT_LAT,
            "lon": DEFAULT_LON,
            "spatial_ref": 0,
        },
    )
