from datetime import datetime, timedelta

import pandas as pd
import xarray as xr
from pydantic import BaseModel, ConfigDict, create_model
from xcube.core.store import DataStore

from irrigation_processor.constants import LOG
from irrigation_processor.core import XcubeDataStoreStorage
from irrigation_processor.core.pipeline import StepRegistry


def split_date_range(start_date, end_date, num_days=30):
    if isinstance(start_date, str):
        start_date = datetime.strptime(start_date, "%Y-%m-%d")
    if isinstance(end_date, str):
        end_date = datetime.strptime(end_date, "%Y-%m-%d")

    result = []
    current = start_date

    while current <= end_date:
        chunk_end = min(current + timedelta(days=num_days - 1), end_date)
        result.append((current.strftime("%Y-%m-%d"), chunk_end.strftime("%Y-%m-%d")))
        current = chunk_end + timedelta(days=1)

    return result


def convert_m_to_mm(dataarray, update_long_name=True):
    converted = dataarray * 1000

    converted.attrs = dataarray.attrs.copy()

    converted.attrs["units"] = "mm"
    converted.attrs["GRIB_units"] = "mm"

    if update_long_name:
        if "long_name" in converted.attrs:
            converted.attrs["long_name"] = (
                converted.attrs["long_name"] + " (millimeters)"
            )
        else:
            converted.attrs["long_name"] = "Potential evaporation (millimeters)"

    return converted


def inject_dynamic_context_from_config(
    config: dict, registry: StepRegistry, storage: XcubeDataStoreStorage
):
    steps = registry.all()

    unknown_steps = (
        set(config) - {"base", "dask", "storage"} - set([step.name for step in steps])
    )

    if unknown_steps:
        raise ValueError(f"Unknown steps in config: {unknown_steps}")

    base_cfg = config.get("base", {})
    if not isinstance(base_cfg, dict):
        raise TypeError("'base' config must be a dict")

    dask_cfg = config.get("dask", {})
    if not isinstance(dask_cfg, dict):
        raise TypeError("'dask' config must be a dict")

    storage_cfg = config.get("storage", {})
    if not isinstance(storage_cfg, dict):
        raise TypeError("'storage' config must be a dict")

    for step_meta in steps:
        step_name = step_meta.name
        step_cfg = config.get(step_name, {})

        if not isinstance(step_cfg, dict):
            raise TypeError(f"Config for step '{step_name}' must be a dict")

        merged_cfg = {
            **base_cfg,
            **dask_cfg,
            **step_cfg,
            **storage_cfg,
            "store": storage.store,
        }

        config_model = create_model(
            f"{step_name.capitalize()}Config",
            __config__=ConfigDict(arbitrary_types_allowed=True),
            **{k: (type(v), v) for k, v in merged_cfg.items()},
        )

        context_obj = config_model(**merged_cfg)
        step_meta.context_cls = lambda obj=context_obj: obj


def get_existing_data(
    *,
    store: DataStore,
    data_id: str,
    load: bool = False,
) -> str | xr.Dataset | None:
    if data_id not in store.list_data_ids():
        return None

    if load:
        return store.open_data(data_id)
    LOG.info(f"Data already exists at {data_id}")
    return data_id


def validate_dataset(context: BaseModel, dataset: xr.Dataset) -> None:
    if not {"lat", "lon", "time"}.issubset(dataset.coords):
        raise ValueError("Dataset must contain 'lat', 'lon', and 'time' coordinates.")

    bbox: list[float] = context.bbox
    min_lon, min_lat, max_lon, max_lat = bbox

    ds_min_lat = float(dataset.lat.min())
    ds_max_lat = float(dataset.lat.max())
    ds_min_lon = float(dataset.lon.min())
    ds_max_lon = float(dataset.lon.max())

    if dataset.lat[0] < dataset.lat[-1]:
        raise ValueError("Latitude must be descending for slicing logic.")

    lat_res = abs(float(dataset.lat.diff("lat").mean().item()))
    lon_res = abs(float(dataset.lon.diff("lon").mean().item()))

    lat_tol = lat_res
    lon_tol = lon_res

    # Latitude check
    if ds_min_lat > min_lat + lat_tol or ds_max_lat < max_lat - lat_tol:
        raise ValueError(
            f"Dataset latitude range [{ds_min_lat}, {ds_max_lat}] "
            f"does not sufficiently cover requested range [{min_lat}, {max_lat}]."
        )

    # Longitude check
    if ds_min_lon > min_lon + lon_tol or ds_max_lon < max_lon - lon_tol:
        raise ValueError(
            f"Dataset longitude range [{ds_min_lon}, {ds_max_lon}] "
            f"does not sufficiently cover requested range [{min_lon}, {max_lon}]."
        )

    # Temporal check
    ds_min_time = dataset.time.min().values
    ds_max_time = dataset.time.max().values

    req_start = pd.to_datetime(context.time_range[0]).to_datetime64()
    req_end = pd.to_datetime(context.time_range[1]).to_datetime64()

    if ds_min_time > req_start or ds_max_time < req_end:
        raise ValueError(
            f"Dataset time range [{ds_min_time}, {ds_max_time}] "
            f"does not fully cover requested range [{req_start}, {req_end}]."
        )

    subset = dataset.sel(
        lat=slice(max_lat, min_lat),
        lon=slice(min_lon, max_lon),
    )

    if subset.lat.size == 0 or subset.lon.size == 0:
        raise ValueError("Spatial subset returned no data.")

    subset = subset.sel(
        time=slice(req_start, req_end),
    )

    if subset.time.size == 0:
        raise ValueError("Temporal subset returned no data.")
