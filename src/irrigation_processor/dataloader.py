import json
import os.path

import xarray as xr
from pydantic import BaseModel
from xcube.core.store import new_data_store
from zappend.api import zappend

from irrigation_processor.constants import (
    CLMS_DATA_ID,
    ERA5_DATA_ID,
    INPUT_DIR,
    LC_DATA_ID,
    logger,
    CDS_VARIABLES,
)
from irrigation_processor.utils import split_date_range

store = new_data_store("file", root=INPUT_DIR)


def load_data(context: BaseModel) -> dict:
    logger.info("loading data...")

    era5_data_id = _get_cds_data(context)
    sm_cube = _get_clms_data(context)
    lc_cube = _get_lc_data(context)

    logger.info("data loaded...")
    return {
        CLMS_DATA_ID: sm_cube,
        LC_DATA_ID: lc_cube,
        "era5_data_id": era5_data_id
            }


def _get_cds_data(context: BaseModel) -> str:
    data_ids = store.list_data_ids()
    if ERA5_DATA_ID in data_ids:
        logger.info(f"CLMS data already exists at {INPUT_DIR}/{ERA5_DATA_ID}")
        return ERA5_DATA_ID

    time_range = context.time_range
    bbox = context.bbox
    data_id = context.cds_data_id
    spatial_res = context.cds_spatial_res

    time_ranges = split_date_range(time_range[0], time_range[1], 5)

    era_store = new_data_store("file", root="era5")
    cds_store = new_data_store("cds", normalize_names=True)

    for _time_range in time_ranges:
        cds_cube = cds_store.open_data(
            data_id,
            cds_store.get_data_opener_ids()[0],
            variable_names=CDS_VARIABLES,
            bbox=bbox,
            spatial_res=spatial_res,
            time_range=_time_range,
        )
        era_store.write_data(
            cds_cube,
            f"era5-{_time_range[0].replace('-', '_')}-{_time_range[1].replace('-', '_')}.zarr",
            replace=True,
        )

    data_ids = era_store.list_data_ids()
    target_dir = f"{INPUT_DIR}/{ERA5_DATA_ID}"
    if not os.path.exists(target_dir):
        os.makedirs(target_dir, exist_ok=True)
    config = {
        "target_dir": target_dir,
        "force_new": True,
        "logging": True,
        "excluded_variables": ["expver", "number"],
    }
    zappend((f"era5/{data_id}" for data_id in sorted(data_ids)), config=config)

    return ERA5_DATA_ID


def _get_clms_data(context: BaseModel) -> xr.Dataset:
    data_ids = store.list_data_ids()
    if CLMS_DATA_ID in data_ids:
        logger.info(f"CLMS data already exists at {INPUT_DIR}/{CLMS_DATA_ID}")
        return store.open_data(CLMS_DATA_ID)
    logger.info("Downloading CLMS Soil Moisture dataset...")

    time_range = context.time_range

    json_file_path = "credentials.json"
    with open(json_file_path, "r") as j:
        credentials = json.loads(j.read())

    clms_data_store = new_data_store("clms", credentials=credentials)

    clms_data = clms_data_store.open_data(
        "daily-surface-soil-moisture-v1.0", time_range=time_range
    )

    clms_ssm_only = clms_data.drop_vars("ssm_noise")

    return clms_ssm_only


def _get_lc_data(context: BaseModel) -> xr.Dataset:
    data_ids = store.list_data_ids()
    if LC_DATA_ID in data_ids:
        logger.info(f"LandCover data already exists at {INPUT_DIR}"
                    f"/{LC_DATA_ID}")
        return store.open_data(LC_DATA_ID)

    time = context.lc_time

    store_lccs = new_data_store(
        "s3", root="deep-esdl-public", storage_options=dict(anon=True)
    )
    mlds_lc = store_lccs.open_data("LC-1x2025x2025-2.0.0.levels")

    lc = mlds_lc.base_dataset
    lc = lc.sel(time=time)
    lc = lc[["crs", "lccs_class"]]

    return lc
