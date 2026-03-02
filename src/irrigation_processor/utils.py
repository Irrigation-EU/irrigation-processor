from datetime import datetime, timedelta
from typing import Any

import pandas as pd
import xarray as xr

from irrigation_processor.config import AppConfig
from irrigation_processor.constants import LOG
from irrigation_processor.core.storage import Storage


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


def get_existing_data(
    *,
    storage: Storage,
    data_id: str,
    load: bool = False,
) -> str | Any | None:
    if not storage.exists(data_id):
        return None

    if load:
        return storage.load(data_id)

    LOG.info(f"Data already exists for id {data_id}")
    return data_id


def validate_dataset(context: AppConfig, dataset: xr.Dataset) -> None:
    if not {"lat", "lon", "time"}.issubset(dataset.coords):
        raise ValueError("Dataset must contain 'lat', 'lon', and 'time' coordinates.")

    bbox: list[float] = context.base.bbox
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
    ds_min_date = dataset.time.min().values.astype("datetime64[D]")
    ds_max_date = dataset.time.max().values.astype("datetime64[D]")

    req_start = (
        pd.to_datetime(context.base.time_range[0])
        .to_datetime64()
        .astype("datetime64[D]")
    )
    req_end = (
        pd.to_datetime(context.base.time_range[1])
        .to_datetime64()
        .astype("datetime64[D]")
    )

    if ds_min_date > req_start or ds_max_date < req_end:
        raise ValueError(
            f"Dataset date range [{ds_min_date}, {ds_max_date}] "
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
