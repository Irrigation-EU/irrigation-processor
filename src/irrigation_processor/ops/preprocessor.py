import numpy as np
import pandas as pd
import xarray as xr
from xcube.core.chunk import chunk_dataset
from xcube_resampling.gridmapping import GridMapping
from xcube_resampling.spatial import resample_in_space

from irrigation_processor.config import AppConfig
from irrigation_processor.constants import (INPUT_FOR_CALIBRATION_ID, LOG,
                                            PROCESSED_CLMS_DATA_ID)
from irrigation_processor.core.storage import Storage
from irrigation_processor.utils import (convert_m_to_mm, get_existing_data,
                                        validate_dataset)


def irrigation_preprocessor(
    context: AppConfig,
    storage: Storage,
    sm_data_id: str,
    lc_data_id: str,
    era5_vars_data_id: str,
    gleam_data_id: str | None,
) -> xr.Dataset:
    result = get_existing_data(
        storage=storage,
        data_id=INPUT_FOR_CALIBRATION_ID,
        load=True,
    )
    if result is not None:
        assert isinstance(result, xr.Dataset)
        validate_dataset(context, result)
        return result

    preprocessed_sm = _soil_moisture_preprocessor(context, storage, sm_data_id)
    preprocessed_lc = _land_cover_preprocessor(context, storage, lc_data_id)
    preprocessed_era5 = _era5_preprocessor(context, storage, era5_vars_data_id)
    preprocessed_gleam = _gleam_preprocessor(context, storage, gleam_data_id)

    merged_ds = _resample_and_merge(
        preprocessed_sm,
        preprocessed_lc,
        preprocessed_era5,
        preprocessed_gleam,
        chunk_sizes=context.preprocessing.merged_chunks.to_dict(),
    )

    LOG.info("preprocessing complete...")
    return merged_ds


def _soil_moisture_preprocessor(
    context: AppConfig, storage: Storage, sm_data_id: str
) -> xr.Dataset:
    result = get_existing_data(
        storage=storage,
        data_id=PROCESSED_CLMS_DATA_ID,
        load=True,
    )
    if result is not None:
        assert isinstance(result, xr.Dataset)
        return result

    clms_data = storage.load(sm_data_id)
    validate_dataset(context, clms_data)
    bbox: list[float] = context.base.bbox

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

    SWI = chunk_dataset(SWI, chunk_sizes=context.preprocessing.swi_chunks.to_dict())

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


def _land_cover_preprocessor(
    context: AppConfig, storage: Storage, lc_data_id: str
) -> xr.DataArray:
    lc_cube = storage.load(lc_data_id)

    keep_classes = context.preprocessing.lc_keep_classes

    lc_binary = lc_cube.lccs_class.isin(keep_classes)

    LOG.info("preprocessed land cover...")
    return lc_binary


def _era5_preprocessor(
    context: AppConfig, storage: Storage, cds_data_id: str
) -> xr.Dataset:
    cds_cube = storage.load(cds_data_id)
    validate_dataset(context, cds_cube)

    if context.base.use_gleam:
        cds_cube = cds_cube.drop_vars("pev")
    else:
        cds_cube["pev"] = cds_cube["pev"] * -1
        cds_cube["pev"] = convert_m_to_mm(cds_cube["pev"])
    cds_cube["tp"] = convert_m_to_mm(cds_cube["tp"])

    LOG.info("preprocessed era5...")
    return cds_cube


def _gleam_preprocessor(
    context: AppConfig, storage: Storage, gleam_data_id: str | None
) -> xr.Dataset | None:
    if gleam_data_id is None:
        return None

    gleam_cube = storage.load(gleam_data_id)
    gleam_cube = gleam_cube.rename({"Ep": "pev"})
    bbox: list[float] = context.base.bbox

    return gleam_cube.sel(lat=slice(bbox[3], bbox[1]), lon=slice(bbox[0], bbox[2]))


def _resample_and_merge(
    soil_moisture: xr.Dataset,
    lc: xr.DataArray,
    era5: xr.Dataset,
    gleam: xr.Dataset | None = None,
    chunk_sizes: dict[str, int] | None = None,
) -> xr.Dataset:
    LOG.info("resampling...")
    gm_sm = GridMapping.from_dataset(soil_moisture)

    cds_in_gm_sm = resample_in_space(era5, target_gm=gm_sm)
    cds_in_gm_sm = cds_in_gm_sm.assign_coords(time=era5.time)

    lc_in_gm_sm = resample_in_space(
        lc.to_dataset(name="lc_binary"), target_gm=gm_sm, agg_methods="mode"
    )
    if "time" in lc_in_gm_sm.dims:
        lc_in_gm_sm = lc_in_gm_sm.squeeze("time", drop=True)

    cds_masked = cds_in_gm_sm.where(lc_in_gm_sm.lc_binary)
    soil_moisture_masked = soil_moisture.where(lc_in_gm_sm.lc_binary)

    cds_masked_aligned = cds_masked.assign_coords(time=soil_moisture_masked.time)

    if gleam is not None:
        gleam_in_gm_sm = resample_in_space(gleam, target_gm=gm_sm)
        gleam_masked = gleam_in_gm_sm.where(lc_in_gm_sm.lc_binary)

        LOG.info("merging along with gleam...")
        ds_combined = xr.merge([soil_moisture_masked, cds_masked_aligned, gleam_masked], join='exact')

    else:
        LOG.info("merging...")
        ds_combined = xr.merge([soil_moisture_masked, cds_masked_aligned], join='exact')

    chunked_ds = chunk_dataset(
        ds_combined,
        chunk_sizes=chunk_sizes or {"time": -1, "lat": 50, "lon": 50},
        format_name="zarr",
    )

    return chunked_ds
