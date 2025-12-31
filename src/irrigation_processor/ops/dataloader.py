import json
import os.path
import time

import xarray as xr
from pydantic import BaseModel
from xcube.core.store import new_data_store, DataStore
from xcube.core.chunk import chunk_dataset
from zappend.api import zappend

from irrigation_processor.constants import (
    CLMS_DATA_ID,
    ERA5_DATA_ID,
    OUTPUT_DIR,
    LC_DATA_ID,
    LOG,
)
from irrigation_processor.utils import split_date_range

def load_data(context: BaseModel) -> dict:
    LOG.info("loading data...")

    era5_data_id = _get_cds_data(context)
    sm_data_id = _get_clms_data(context)
    lc_cube = _get_lc_data(context)

    LOG.info("data loaded...")
    return {
        "sm_data_id": sm_data_id,
        # the below two items that are returned are actual xarray datasets
        # which are then handled by the framework to write it to and load
        # from the disk. The downstream tasks can refer to this outputs using
        # FromTask("dataloader", CLMS_DATA_ID). What this would do is that
        # internally, this would store the data and pass the stored paths,
        # but to the user it looks like we are passing Datasets directly.
        # This makes it Airflow compatible.
        LC_DATA_ID: lc_cube,
        "era5_data_id": era5_data_id,  # Here, we can pass
        # a string as key that its dependencies must refer to when they want
        # to use this output as their input in FromTask class. e.g. FromTask(
        # "dataloader", "era5_data_id").
    }


def _get_cds_data(context: BaseModel) -> str:
    store: DataStore = context.store
    data_ids = store.list_data_ids()
    if ERA5_DATA_ID in data_ids:
        LOG.info(f"CDS data already exists at {OUTPUT_DIR}/{ERA5_DATA_ID}")
        return ERA5_DATA_ID

    time_range = context.time_range
    bbox = context.bbox
    data_id = context.cds_data_id
    spatial_res = context.cds_spatial_res
    variables_name = context.cds_variable_names

    time_ranges = split_date_range(time_range[0], time_range[1], 5)

    CDS_SUBDIR = "era5"
    cds_store = new_data_store("cds", normalize_names=True)

    for _time_range in time_ranges:
        cds_cube = cds_store.open_data(
            data_id,
            cds_store.get_data_opener_ids()[0],
            variable_names=variables_name,
            bbox=bbox,
            spatial_res=spatial_res,
            time_range=_time_range,
        )
        LOG.info(f"Writing CDS data for time range: {_time_range}")
        store.write_data(
            cds_cube,
            f"{CDS_SUBDIR}/era5-{_time_range[0].replace('-', '_')}-{_time_range[1].replace('-', '_')}.zarr",
            replace=True,
        )

    all_data_ids = store.list_data_ids()
    data_ids = sorted([data_id for data_id in all_data_ids
                       if (f"{CDS_SUBDIR}/" in data_id)])

    def get_dataset(data_id: str):
        ds = store.open_data(data_id)
        ds = ds.drop_vars(["expver", "number"])
        pev_daily = (
            ds["pev"].resample(time="1D").mean()
        )
        tp_daily = (
            ds["tp"].resample(time="1D").mean()
        )
        ### Do we need to take mean() or last() instead, ask Jacopo???

        merged_ds = xr.merge([pev_daily, tp_daily])
        return merged_ds

    datasets = []
    for data_id in sorted(data_ids):
        datasets.append(get_dataset(data_id))

    if context.cds_optimize_writing:
        LOG.info("Writing CDS data faster...")
        LOG.info("Concatenating CDS data...")
        ds = xr.concat(datasets, dim="time", join="left")
        LOG.info("Chunking CDS data...")
        ds = chunk_dataset(ds, {"time": 10, "lat": 178, "lon": 306}, format_name="zarr")
        LOG.info("Writing chunked CDS data...")
        store.write_data(ds, "era5_chunked.zarr")

        ds = store.open_data("era5_chunked.zarr")
        LOG.info("Rechunking CDS data to make it time optimized...")
        ds = chunk_dataset(ds, {"time": -1, "lat": 50, "lon": 50}, format_name="zarr")
        LOG.info("Writing final CDS data...")
        store.write_data(ds, ERA5_DATA_ID)
        LOG.info("Deleting chunked CDS data...")
        store.delete_data("era5_chunked.zarr")
    else:
        if store.protocol == "s3":
            LOG.info("Using S3 storage")
            storage_options = {
                "anon": False,
                "key": os.getenv("XCUBE_AWS_ACCESS_KEY_ID"),
                "secret": os.getenv("XCUBE_AWS_SECRET_ACCESS_KEY"),
                "client_kwargs": {
                    "endpoint_url": os.getenv("XCUBE_AWS_ENDPOINT_URL")
                }
            }
            target_path = f"s3://{os.getenv("XCUBE_BUCKET_NAME")}/{ERA5_DATA_ID}"
        else:
            LOG.info("Using file storage")
            storage_options = {}
            target_path = f"{OUTPUT_DIR}/{ERA5_DATA_ID}"
        total_time_steps = sum(ds.sizes["time"] for ds in datasets)
        config = {
            "target_dir": target_path,
            "target_storage_options": storage_options,
            "force_new": True,
            "logging": True,
            "excluded_variables": ["expver", "number"],
            "append_dim": "time",
            "variables": {
                "*": {"encoding": {"chunks": None}},
                "time": {"dims": ["time"], "encoding": {"chunks": [total_time_steps]}},
                "pev": {
                    "dims": ["time", "lat", "lon"],
                    "encoding": {
                        "chunks": [total_time_steps, 15, 15],
                        "dtype": "float32",
                    },
                },
                "tp": {
                    "dims": ["time", "lat", "lon"],
                    "encoding": {
                        "chunks": [total_time_steps, 15, 15],
                        "dtype": "float32",
                    },
                },
            },
        }
        LOG.info("zappending now...")
        zappend(data_ids, slice_source=get_dataset, config=config)

    return ERA5_DATA_ID


def _get_clms_data(context: BaseModel) -> str:
    store: DataStore = context.store
    data_ids = store.list_data_ids()
    if CLMS_DATA_ID in data_ids:
        LOG.info(f"CLMS data already exists at {OUTPUT_DIR}/{CLMS_DATA_ID}")
        return CLMS_DATA_ID
    LOG.info("Downloading CLMS Soil Moisture dataset...")

    time_range: list = context.time_range

    time_ranges = split_date_range(time_range[0], time_range[1], 5)

    json_file_path = "credentials.json"
    with open(json_file_path, "r") as j:
        credentials = json.loads(j.read())

    clms_store = new_data_store("clms", credentials=credentials)
    CLMS_SUBDIR = "clms"

    for i, _time_range in enumerate(time_ranges):
        while True:
            try:
                LOG.info(f"Reading time_range: {_time_range}")
                clms_data = clms_store.open_data(
                    "daily-surface-soil-moisture-v1.0", time_range=_time_range
                )

                filename = (f"{CLMS_SUBDIR}/clms_sm-{_time_range[0].replace('-', '_')}"
                            f"-{_time_range[1].replace('-', '_')}.zarr")
                clms_data = clms_data.rename({"x": "lon", "y": "lat"})

                LOG.info("Writing data...")
                store.write_data(clms_data, filename, replace=True)

                LOG.info(f"Done: {_time_range}")

                # force refresh of clms data store
                if i % 10 == 0:
                    clms_store = new_data_store("clms", credentials=credentials)
                time.sleep(20)
                break
            except Exception as e:
                LOG.error(f"Exception for {_time_range}: {e}")
                LOG.info("Waiting 45 seconds before retrying...")
                time.sleep(45)

    def get_dataset(data_id: str):
        ds = store.open_data(data_id)
        ds = ds.drop_vars(["ssm_noise"])
        ds["ssm"] = ds["ssm"].astype("float32")
        return ds

    all_data_ids = store.list_data_ids()
    data_ids = sorted([data_id for data_id in all_data_ids if f"{CLMS_SUBDIR}/"
                       in data_id])
    datasets = []
    [
        datasets.append(
            store.open_data(data_id)
        )
        for data_id in sorted(data_ids)
    ]
    total_time_steps = sum(ds.sizes["time"] for ds in datasets)

    if store.protocol == "s3":
        LOG.info("Using S3 storage")
        storage_options = {
            "anon": False,
            "key": os.getenv("XCUBE_AWS_ACCESS_KEY_ID"),
            "secret": os.getenv("XCUBE_AWS_SECRET_ACCESS_KEY"),
            "client_kwargs": {"endpoint_url": os.getenv("XCUBE_AWS_ENDPOINT_URL")},
        }
        target_path = f"s3://{os.getenv('XCUBE_BUCKET_NAME')}/{CLMS_DATA_ID}"
    else:
        LOG.info("Using file storage")
        storage_options = {}
        target_path = f"{OUTPUT_DIR}/{CLMS_DATA_ID}"

    config = {
        "target_dir": target_path,
        "target_storage_options": storage_options,
        "force_new": True,
        "logging": True,
        "excluded_variables": ["expver", "number"],
        "append_dim": "time",
        "variables": {
            "*": {"encoding": {"chunks": None}},
            "time": {"dims": ["time"], "encoding": {"chunks": [total_time_steps]}},
            "ssm": {
                "dims": ["time", "lat", "lon"],
                "encoding": {
                    "chunks": [total_time_steps, 150, 150],
                    "dtype": "float32",
                },
            },
        },
    }
    LOG.info("zappending now...")
    zappend(data_ids, slice_source=get_dataset, config=config)

    return CLMS_DATA_ID


def _get_lc_data(context: BaseModel) -> xr.Dataset:
    store: DataStore = context.store
    data_ids = store.list_data_ids()
    if LC_DATA_ID in data_ids:
        LOG.info(f"LandCover data already exists at {OUTPUT_DIR}"
                    f"/{LC_DATA_ID}")
        return store.open_data(LC_DATA_ID)
    LOG.info("Downloading LandCover dataset from S3...")
    time = context.lc_time

    store_lccs = new_data_store(
        "s3", root="deep-esdl-public", storage_options=dict(anon=True)
    )
    mlds_lc = store_lccs.open_data("LC-1x2025x2025-2.0.0.levels")

    lc = mlds_lc.base_dataset
    lc = lc.sel(time=time)
    lc = lc[["crs", "lccs_class"]]

    return lc
