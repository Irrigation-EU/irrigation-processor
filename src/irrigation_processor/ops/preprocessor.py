import numpy as np
import pandas as pd
import xarray as xr
from pydantic import BaseModel
from xcube.core.chunk import chunk_dataset
from xcube.core.store import DataStore
from xcube_resampling.gridmapping import GridMapping
from xcube_resampling.spatial import resample_in_space

from irrigation_processor.constants import (
    INPUT_FOR_CALIBRATION_ID,
    LOG,
    OUTPUT_DIR,
    PROCESSED_CLMS_DATA_ID,
)
from irrigation_processor.utils import convert_m_to_mm

def irrigation_preprocessor(
    context: BaseModel, sm_data_id: str, lc_cube: xr.Dataset, era5_data_id: str
) -> xr.Dataset:
    store: DataStore = context.store
    data_ids = store.list_data_ids()

    if INPUT_FOR_CALIBRATION_ID in data_ids:
        LOG.info(
            "Irrigation inputs are already preprocessed with "
            f"data_id: {INPUT_FOR_CALIBRATION_ID}"
        )
        return store.open_data(INPUT_FOR_CALIBRATION_ID)

    preprocessed_sm = _soil_moisture_preprocessor(context, sm_data_id)
    preprocessed_lc = _land_cover_preprocessor(context, lc_cube)
    preprocessed_era5 = _era5_preprocessor(context, era5_data_id)

    merged_ds = _resample_and_merge(preprocessed_sm, preprocessed_lc, preprocessed_era5)

    LOG.info("preprocessing complete...")

    # here we return the dataset as is. So, we can
    # specify it in the output section of the registry for this step with a
    # name that needs to be used by downstream tasks. If no output is
    # described, it will raise an error.
    return merged_ds


def _soil_moisture_preprocessor(context: BaseModel, sm_data_id: str) -> xr.Dataset:
    store: DataStore = context.store
    data_ids = store.list_data_ids()
    if PROCESSED_CLMS_DATA_ID in data_ids:
        LOG.info(
            f"CLMS processed data already exists at {OUTPUT_DIR}"
            f"/{PROCESSED_CLMS_DATA_ID}"
        )
        return store.open_data(PROCESSED_CLMS_DATA_ID)

    clms_data = store.open_data(sm_data_id)
    bbox = context.bbox

    # Interpolation
    full_time = pd.date_range(
        start=clms_data.time.min().item(),
        end=clms_data.time.max().item(),
        freq="D",
    )
    clms_data = clms_data.reindex(time=full_time)
    clms_data = clms_data.sel(lat=slice(bbox[3], bbox[1]), lon=slice(bbox[0], bbox[2]))

    julian_dates = xr.DataArray(
        pd.to_datetime(clms_data.time.values).to_julian_date().values,
        dims="time",
        coords={"time": clms_data.time},
    )

    sm_filled = clms_data["ssm"].interpolate_na(dim="time", method="linear")

    sm_clipped = sm_filled.clip(max=100)

    sm_normalized = (sm_clipped - sm_clipped.min(dim="time")) / (
        sm_clipped.max(dim="time") - sm_clipped.min(dim="time")
    )

    SWI = xr.apply_ufunc(
        _swicomp_nan,
        sm_normalized,
        julian_dates,
        input_core_dims=[["time"], ["time"]],
        output_core_dims=[["time"]],
        vectorize=True,
        dask="parallelized",
        kwargs={"ctime": 2},
        output_dtypes=[clms_data.ssm.dtype],
    )

    SWI = SWI.transpose("time", "lat", "lon")

    SWI = chunk_dataset(SWI, chunk_sizes={"time": -1, "lat": 128, "lon": 128})

    LOG.info("preprocessed soil moisture...")

    return SWI.to_dataset(name="SWI")


def _swicomp_nan(in_data, in_jd, ctime=2):
    filtered = np.empty(len(in_data))
    gain = 1
    filtered.fill(np.nan)

    ID = np.where(~np.isnan(in_data))
    if len(ID[0]) == 0:
        return filtered

    D = in_jd[ID]
    SWI = in_data[ID].copy()
    tdiff = np.diff(D)

    for i in range(2, SWI.size):
        gain = gain / (gain + np.exp(-tdiff[i - 1] / ctime))
        SWI[i] = SWI[i - 1] + gain * (SWI[i] - SWI[i - 1])

    filtered[ID] = SWI
    return filtered


def _land_cover_preprocessor(context: BaseModel, lc: xr.Dataset) -> xr.DataArray:
    bbox = context.bbox
    lc_subset = lc.sel(lat=slice(bbox[3], bbox[1]), lon=slice(bbox[0], bbox[2]))

    keep_classes = [
        10,
        11,
        12,
        20,
        30,
    ]  # These are classes in LandCover related to Croplands

    filtered_lc = lc_subset.lccs_class.where(lc_subset["lccs_class"].isin(keep_classes))

    keep_classes_np = np.array(keep_classes, dtype=filtered_lc.dtype)
    LOG.info("preprocessed land cover...")

    return filtered_lc.isin(keep_classes_np).astype("uint8")


def _era5_preprocessor(context: BaseModel, cds_data_id: str) -> xr.Dataset:
    store: DataStore = context.store
    cds_cube = store.open_data(cds_data_id)

    cds_cube["pev"] = cds_cube["pev"] * -1
    cds_cube["pev"] = convert_m_to_mm(cds_cube["pev"])
    cds_cube["tp"] = convert_m_to_mm(cds_cube["tp"])

    LOG.info("preprocessed era5...")
    return cds_cube


def _resample_and_merge(
    soil_moisture: xr.Dataset, lc: xr.DataArray, era5: xr.Dataset
) -> xr.Dataset:
    LOG.info("resampling...")
    gm_sm = GridMapping.from_dataset(soil_moisture)

    cds_in_gm_sm = resample_in_space(era5, target_gm=gm_sm)
    cds_in_gm_sm = cds_in_gm_sm.assign_coords(time=era5.time)

    lc.attrs["flag_values"] = [0, 1]
    lc.attrs["flag_meanings"] = "no_data croplands_land_cover"
    lc_in_gm_sm = resample_in_space(
        lc.to_dataset(name="lc_binary"), target_gm=gm_sm, agg_methods="mode"
    )
    lc_in_gm_sm = lc_in_gm_sm.drop_vars("time")

    cds_masked = cds_in_gm_sm.where(lc_in_gm_sm.lc_binary == 1)
    soil_moisture_masked = soil_moisture.where(lc_in_gm_sm.lc_binary == 1)

    cds_masked_aligned = cds_masked.assign_coords(time=soil_moisture_masked.time)

    LOG.info("merging...")
    ds_combined = xr.merge([soil_moisture_masked, cds_masked_aligned])

    chunked_ds = chunk_dataset(
        ds_combined,
        chunk_sizes={"time": -1, "lat": 50, "lon": 50},
        format_name="zarr",
    )

    return chunked_ds
