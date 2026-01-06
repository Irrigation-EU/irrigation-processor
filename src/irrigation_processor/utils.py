from datetime import datetime, timedelta

import xarray as xr
from pydantic import ConfigDict, create_model

from irrigation_processor.constants import LOG, OUTPUT_DIR
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

    unknown_steps = set(config) - {"base"} - set([step.name for step in steps])
    if unknown_steps:
        raise ValueError(f"Unknown steps in config: {unknown_steps}")

    base_cfg = config.get("base", {})
    if not isinstance(base_cfg, dict):
        raise TypeError("'base' config must be a dict")

    for step_meta in steps:
        step_name = step_meta.name
        step_cfg = config.get(step_name, {})

        if not isinstance(step_cfg, dict):
            raise TypeError(f"Config for step '{step_name}' must be a dict")

        merged_cfg = {**base_cfg, **step_cfg, "store": storage.store}

        config_model = create_model(
            f"{step_name.capitalize()}Config",
            __config__=ConfigDict(arbitrary_types_allowed=True),
            **{k: (type(v), v) for k, v in merged_cfg.items()},
        )

        context_obj = config_model(**merged_cfg)
        step_meta.context_cls = lambda obj=context_obj: obj


def get_existing_data(
    *,
    store,
    data_id: str,
    load: bool = False,
) -> str | xr.Dataset | None:
    if data_id not in store.list_data_ids():
        return None

    if load:
        return store.open_data(data_id)
    LOG.info(f"Data already exists at {OUTPUT_DIR}/{data_id}")
    return data_id
