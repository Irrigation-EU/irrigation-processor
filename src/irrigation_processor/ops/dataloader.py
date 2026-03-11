import json
import os.path
import time

import xarray as xr
from xcube.core.chunk import chunk_dataset
from xcube.core.store import new_data_store
from zappend.api import zappend

from irrigation_processor.config import AppConfig
from irrigation_processor.constants import (CLMS_DATA_ID, ERA5_DATA_ID,
                                            GLEAM_DATA_ID, LC_DATA_ID, LOG)
from irrigation_processor.core.storage import Storage
from irrigation_processor.utils import get_existing_data, split_date_range


def load_data(context: AppConfig, storage: Storage) -> dict[str, str | None]:
    LOG.info("loading data...")

    era5_data_id = _get_cds_data(context, storage)
    sm_data_id = _get_clms_data(context, storage)
    lc_data_id = _get_lc_data(context, storage)

    results: dict[str, str | None] = {
        "sm_data_id": sm_data_id,
        "lc_data_id": lc_data_id,
        "era5_vars_data_id": era5_data_id,
    }

    if context.base.use_gleam:
        gleam_data_id = _get_gleam_data(context, storage)
        results["gleam_data_id"] = gleam_data_id
    else:
        results["gleam_data_id"] = None

    LOG.info("data loaded...")
    return results


def _get_cds_data(context: AppConfig, storage: Storage) -> str:
    output_dir: str = context.storage.store_kwargs.root

    result = get_existing_data(storage=storage, data_id=ERA5_DATA_ID)

    if result is not None:
        assert isinstance(result, str)
        return result

    time_range: list[str] = context.base.time_range
    bbox: list[float] = list(context.base.bbox)
    spatial_res: float = context.dataloader.cds_spatial_res
    bbox[0] = bbox[0] - spatial_res
    bbox[1] = bbox[1] - spatial_res
    bbox[2] = bbox[2] + spatial_res
    bbox[3] = bbox[3] + spatial_res

    data_id: str = context.dataloader.cds_data_id
    variables_name: list[str] = context.dataloader.cds_variable_names
    if context.base.use_gleam:
        variables_name = [v for v in variables_name if v != "potential_evaporation"]

    LOG.info(f"Variables required from ERA5-Land, {variables_name}")

    time_ranges = split_date_range(time_range[0], time_range[1], 5)

    CDS_SUBDIR = "era5"
    cds_store = new_data_store("cds", normalize_names=True)

    for _time_range in time_ranges:
        filename = f"{CDS_SUBDIR}/era5-{_time_range[0].replace('-', '_')}-{_time_range[1].replace('-', '_')}.zarr"
        if (
            get_existing_data(
                storage=storage,
                data_id=filename,
            )
            is not None
        ):
            continue
        cds_cube = cds_store.open_data(
            data_id,
            cds_store.get_data_opener_ids()[0],
            variable_names=variables_name,
            bbox=bbox,
            spatial_res=spatial_res,
            time_range=_time_range,
        )
        LOG.info(f"Writing CDS data for time range: {_time_range}")
        storage.save(key=filename, obj=cds_cube)

    all_data_ids = storage.list_ids()
    data_ids = sorted(
        [data_id for data_id in all_data_ids if (f"{CDS_SUBDIR}/" in data_id)]
    )

    def get_dataset(data_id: str):
        ds = storage.load(data_id)
        ds = ds.drop_vars(["expver", "number"])

        # taking last() as the variables are accumulated over 24 hours
        # https://confluence.ecmwf.int/display/CKB/ERA5-Land%3A+data+documentation#heading-Accumulations
        pev_daily = ds["pev"].resample(time="1D").last()
        tp_daily = ds["tp"].resample(time="1D").last()

        merged_ds = xr.merge([pev_daily, tp_daily])
        return merged_ds

    datasets = []
    for data_id in sorted(data_ids):
        datasets.append(get_dataset(data_id))

    if context.dataloader.cds_optimize_writing:
        LOG.info("Writing CDS data faster...")
        LOG.info("Concatenating CDS data...")
        ds = xr.concat(datasets, dim="time", join="left")
        LOG.info("Chunking CDS data...")
        ds = chunk_dataset(
            ds, context.dataloader.cds_intermediate_chunks.to_dict(), format_name="zarr"
        )
        LOG.info("Writing chunked CDS data...")
        storage.save(key="era5_chunked.zarr", obj=ds)

        ds = storage.load("era5_chunked.zarr")
        LOG.info("Rechunking CDS data to make it time optimized...")
        ds = chunk_dataset(
            ds, context.dataloader.cds_final_chunks.to_dict(), format_name="zarr"
        )
        LOG.info("Writing final CDS data...")
        storage.save(key=ERA5_DATA_ID, obj=ds)
        LOG.info("Deleting chunked CDS data...")
        storage.delete("era5_chunked.zarr")
    else:
        if storage.protocol == "s3":
            LOG.info("Using S3 storage")
            storage_options = {
                "anon": False,
                "key": os.getenv("XCUBE_AWS_ACCESS_KEY_ID"),
                "secret": os.getenv("XCUBE_AWS_SECRET_ACCESS_KEY"),
                "client_kwargs": {"endpoint_url": os.getenv("XCUBE_AWS_ENDPOINT_URL")},
            }
            target_path = f"s3://{os.getenv('XCUBE_BUCKET_NAME')}/{ERA5_DATA_ID}"
        else:
            LOG.info("Using file storage")
            storage_options = {}
            target_path = f"{output_dir}/{ERA5_DATA_ID}"

        total_time_steps = sum(ds.sizes["time"] for ds in datasets)
        config = {
            "target_dir": target_path,
            "target_storage_options": storage_options,
            "force_new": False,
            "logging": True,
            "excluded_variables": ["expver", "number"],
            "append_dim": "time",
            "variables": {
                "*": {"encoding": {"chunks": None}},
                "time": {"dims": ["time"], "encoding": {"chunks": [total_time_steps]}},
                "pev": {
                    "dims": ["time", "lat", "lon"],
                    "encoding": {
                        "chunks": [total_time_steps]
                        + context.dataloader.cds_zappend_spatial_chunks,
                        "dtype": "float32",
                    },
                },
                "tp": {
                    "dims": ["time", "lat", "lon"],
                    "encoding": {
                        "chunks": [total_time_steps]
                        + context.dataloader.cds_zappend_spatial_chunks,
                        "dtype": "float32",
                    },
                },
            },
        }
        LOG.info("zappending now...")
        zappend(data_ids, slice_source=get_dataset, config=config)

    return ERA5_DATA_ID


def _get_clms_data(context: AppConfig, storage: Storage) -> str:
    output_dir: str = context.storage.store_kwargs.root

    result = get_existing_data(
        storage=storage,
        data_id=CLMS_DATA_ID,
    )

    if result is not None:
        assert isinstance(result, str)
        return result

    LOG.info("Downloading CLMS Soil Moisture dataset...")

    time_range: list[str] = context.base.time_range

    time_ranges = split_date_range(time_range[0], time_range[1], 5)

    json_file_path = "credentials.json"
    with open(json_file_path, "r") as j:
        credentials = json.loads(j.read())

    clms_store = new_data_store("clms", credentials=credentials)
    CLMS_SUBDIR = "clms"

    for i, _time_range in enumerate(time_ranges):
        filename = (
            f"{CLMS_SUBDIR}/clms_sm-{_time_range[0].replace('-', '_')}"
            f"-{_time_range[1].replace('-', '_')}.zarr"
        )

        if (
            get_existing_data(
                storage=storage,
                data_id=filename,
            )
            is not None
        ):
            continue
        for attempt in range(1, 11):
            try:
                LOG.info(f"Reading time_range: {_time_range}")
                clms_data = clms_store.open_data(
                    "daily-surface-soil-moisture-v1.0", time_range=_time_range
                )

                clms_data = clms_data.rename({"x": "lon", "y": "lat"})

                LOG.info("Writing data...")
                storage.save(key=filename, obj=clms_data)

                LOG.info(f"Done: {_time_range}")

                # force refresh of clms data store
                if i % 10 == 0:
                    clms_store = new_data_store("clms", credentials=credentials)
                time.sleep(20)
                break
            except Exception as e:
                LOG.error(f"Exception for {_time_range}: {e}")
                if attempt == 10:
                    raise
                LOG.info("Waiting 45 seconds before retrying...")
                time.sleep(45)

    def get_dataset(data_id: str):
        ds = storage.load(data_id)
        ds = ds.drop_vars(["ssm_noise"])
        ds["ssm"] = ds["ssm"].astype("float32")
        return ds

    all_data_ids = storage.list_ids()
    data_ids = sorted(
        [data_id for data_id in all_data_ids if f"{CLMS_SUBDIR}/" in data_id]
    )
    datasets = []
    for data_id in sorted(data_ids):
        datasets.append(storage.load(data_id))
    total_time_steps = sum(ds.sizes["time"] for ds in datasets)

    if storage.protocol == "s3":
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
        target_path = f"{output_dir}/{CLMS_DATA_ID}"

    config = {
        "target_dir": target_path,
        "target_storage_options": storage_options,
        "force_new": False,
        "logging": True,
        "excluded_variables": ["expver", "number"],
        "append_dim": "time",
        "variables": {
            "*": {"encoding": {"chunks": None}},
            "time": {"dims": ["time"], "encoding": {"chunks": [total_time_steps]}},
            "ssm": {
                "dims": ["time", "lat", "lon"],
                "encoding": {
                    "chunks": [total_time_steps]
                    + context.dataloader.clms_zappend_spatial_chunks,
                    "dtype": "float32",
                },
            },
        },
    }
    LOG.info("zappending now...")
    zappend(data_ids, slice_source=get_dataset, config=config)

    return CLMS_DATA_ID


def _get_lc_data(context: AppConfig, storage: Storage) -> str:
    result = get_existing_data(storage=storage, data_id=LC_DATA_ID)
    if result is not None:
        assert isinstance(result, str)
        return result

    LOG.info("Downloading LandCover dataset from CDS...")

    data_id: str = context.dataloader.lc_data_id
    time_range: list[str] = context.dataloader.lc_time_range
    bbox: list[float] = context.base.bbox

    cds_store = new_data_store("cds", normalize_names=True)
    lc = cds_store.open_data(
        data_id,
        bbox=bbox,
        time_range=time_range,
    )

    lc = lc[["crs", "lccs_class"]]

    storage.save(LC_DATA_ID, lc)

    return LC_DATA_ID


def _get_gleam_data(context: AppConfig, storage: Storage) -> str:
    LOG.info("Loading Gleam dataset from xcube storage...")
    result = get_existing_data(storage=storage, data_id=GLEAM_DATA_ID)
    assert result is not None, "Gleam dataset must be provided locally in zarr format."
    assert isinstance(result, str)
    return result
