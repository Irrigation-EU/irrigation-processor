import json

from xcube.core.store import new_data_store

from zappend.api import zappend

from irrigation_processor.constants import (
    CLMS_DATA_ID,
    ERA5_DATA_ID,
    LC_DATA_ID,
    INPUT_DIR,
    logger,
)
from irrigation_processor.steps import DataLoaderContext
from irrigation_processor.utils import split_date_range

store = new_data_store("file", root=INPUT_DIR)


def load_data(context: DataLoaderContext) -> dict:
    logger.info("loading data..." )
    era5_path = _get_cds_data(context)
    sm_path = _get_clms_data(context)
    lc_path = _get_lc_data(context)
    logger.info(f"data loaded...{sm_path}, {lc_path}, {era5_path}")
    return {"sm_path": sm_path, "lc_path": lc_path, "era5_path": era5_path}


def _get_cds_data(context: DataLoaderContext) -> str:
    data_ids = store.list_data_ids()
    if ERA5_DATA_ID in data_ids:
        return f"{INPUT_DIR}/{ERA5_DATA_ID}"

    time_range = context.time_range
    bbox = context.bbox
    data_id = context.cds_data_id
    variables = context.cds_variables
    spatial_res = context.cds_spatial_res


    time_ranges = split_date_range(time_range[0], time_range[1], 5)

    era_store = new_data_store("file", root="era5")
    cds_store = new_data_store("cds", normalize_names=True)

    for _time_range in time_ranges:
        cds_cube = cds_store.open_data(
            data_id,
            cds_store.get_data_opener_ids()[0],
            variable_names=variables,
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
    config = {
        "target_dir": f"{INPUT_DIR}/{ERA5_DATA_ID}",
        "force_new": True,
        "logging": True,
        "excluded_variables": ["expver", "number"],
    }
    zappend((f"era5/{data_id}" for data_id in sorted(data_ids)), config=config)

    return f"{INPUT_DIR}/{ERA5_DATA_ID}"


def _get_clms_data(context: DataLoaderContext) -> str:
    data_ids = store.list_data_ids()
    if CLMS_DATA_ID in data_ids:
        return f"{INPUT_DIR}/{CLMS_DATA_ID}"

    time_range = context.time_range

    json_file_path = "credentials.json"
    with open(json_file_path, "r") as j:
        credentials = json.loads(j.read())

    clms_data_store = new_data_store("clms", credentials=credentials)

    clms_data = clms_data_store.open_data(
        "daily-surface-soil-moisture-v1.0", time_range=time_range
    )

    clms_ssm_only = clms_data.drop_vars("ssm_noise")

    store.write_data(clms_ssm_only, CLMS_DATA_ID)

    return f"{INPUT_DIR}/{CLMS_DATA_ID}"


def _get_lc_data(context: DataLoaderContext) -> str:
    data_ids = store.list_data_ids()
    if LC_DATA_ID in data_ids:
        return f"{INPUT_DIR}/{LC_DATA_ID}"

    time = context.lc_time

    store_lccs = new_data_store(
        "s3", root="deep-esdl-public", storage_options=dict(anon=True)
    )
    mlds_lc = store_lccs.open_data("LC-1x2025x2025-2.0.0.levels")

    lc = mlds_lc.base_dataset
    lc = lc.sel(time=time)
    lc = lc[["crs", "lccs_class"]]
    store.write_data(lc, LC_DATA_ID)

    return f"{INPUT_DIR}/{LC_DATA_ID}"
