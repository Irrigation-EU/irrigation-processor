import geopandas as gpd
import xarray as xr
import pandas as pd
import numpy as np
from geopandas import GeoDataFrame
from xcube.core.chunk import chunk_dataset
from xcube.core.geom import mask_dataset_by_geometry

from xcube.core.store import new_data_store
from xcube_resampling.gridmapping import GridMapping
from xcube_resampling.spatial import resample_in_space

from irrigation_processor.constants import (
    CLMS_DATA_ID,
    PROCESSED_CLMS_DATA_ID,
    LC_DATA_ID,
    ERA5_DATA_ID,
    INPUT_DIR,
    INPUT_FOR_CALIBRATION_ID,
    logger,
)
from irrigation_processor.steps import PreprocessorContext
from irrigation_processor.utils import convert_m_to_mm


store = new_data_store("file", root=INPUT_DIR)

def irrigation_preprocessor(context: PreprocessorContext, sm_path, lc_path,
                            era5_path) -> dict:
    logger.info("irrigation preprocessor context...")

    data_ids = store.list_data_ids()

    if INPUT_FOR_CALIBRATION_ID in data_ids:
        logger.info("Irrigation inputs are already preprocessed with "
                    f"data_id: {INPUT_FOR_CALIBRATION_ID}")
        return {"preprocessed_path": f"{INPUT_DIR}/{INPUT_FOR_CALIBRATION_ID}"}

    # TODO: Use the paths provided to create the store and data_ids

    spatial_mask_path = context.spatial_mask_path

    gdf = gpd.read_file(spatial_mask_path)

    preprocessed_sm = _soil_moisture_preprocessor(context)
    preprocessed_lc = _land_cover_preprocessor(context)
    preprocessed_era5 = _era5_preprocessor(context)

    merged_path = _merge(preprocessed_sm, preprocessed_lc, preprocessed_era5, gdf)

    logger.info(f"preprocessing complete...{merged_path}")
    return {"preprocessed_path": merged_path}

def _soil_moisture_preprocessor(context: PreprocessorContext) -> xr.Dataset:

    bbox = context.bbox

    clms_data = store.open_data(CLMS_DATA_ID)

    # Interpolation
    full_time = pd.date_range(
        start=clms_data.time.min().item(), end=clms_data.time.max().item(), freq="D"
    )
    clms_data = clms_data.reindex(time=full_time)
    clms_data = clms_data.sel(lat=slice(bbox[3], bbox[1]), lon=slice(bbox[0], bbox[2]))

    julian_dates = xr.DataArray(
        pd.to_datetime(clms_data.time.values).to_julian_date().values,
        dims="time",
        coords={"time": clms_data.time},
    )

    clms_data_chunked = clms_data.chunk({"time": -1})

    sm_filled = clms_data_chunked["ssm"].interpolate_na(
        dim="time", method="linear"
    )

    sm_clipped = sm_filled.clip(max=100)

    sm_normalized = (sm_clipped - sm_clipped.min()) / (
        sm_clipped.max() - sm_clipped.min()
    )

    SWI = xr.apply_ufunc(
        swicomp_nan,
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

    store.write_data(SWI.to_dataset(name="SWI"), PROCESSED_CLMS_DATA_ID)
    logger.info("preprocessed soil moisture...")

    return store.open_data(PROCESSED_CLMS_DATA_ID)



def swicomp_nan(in_data, in_jd, ctime=2):
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
        gain = gain / (gain + np.exp(- tdiff[i - 1] / ctime))
        SWI[i] = SWI[i - 1] + gain * (SWI[i] - SWI[i-1])

    filtered[ID] = SWI
    return filtered


def _land_cover_preprocessor(context: PreprocessorContext) -> xr.Dataset:
    bbox = context.bbox
    lc = store.open_data(LC_DATA_ID)
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
    logger.info("preprocessed land cover...")

    return filtered_lc.isin(keep_classes_np).astype("uint8")


def _era5_preprocessor(context: PreprocessorContext) -> xr.Dataset:
    cds_cube = store.open_data(ERA5_DATA_ID)

    cds_cube["pev"] = cds_cube["pev"] * -1

    pev_daily = cds_cube["pev"].resample(time="1D").last()
    tp_daily = cds_cube["tp"].resample(time="1D").last()

    cds_cube_daily = xr.merge([pev_daily, tp_daily])

    cds_cube_daily["pev"] = convert_m_to_mm(cds_cube_daily["pev"])
    cds_cube_daily["tp"] = convert_m_to_mm(cds_cube_daily["tp"])
    logger.info("preprocessed era5...")

    return cds_cube_daily


def _merge(soil_moisture: xr.Dataset, lc: xr.Dataset, era5: xr.Dataset, gdf: GeoDataFrame) -> str:
    logger.info("merging...")
    gm_sm = GridMapping.from_dataset(soil_moisture)

    cds_in_gm_sm = resample_in_space(era5, target_gm=gm_sm)
    cds_in_gm_sm = cds_in_gm_sm.assign_coords(time=era5.time)

    lc.attrs["flag_values"] = [0, 1]
    lc.attrs["flag_meanings"] = "no_data croplands_land_cover"
    lc_in_gm_sm = resample_in_space(
        lc.to_dataset(name="lc_binary"), target_gm=gm_sm, agg_methods="mode"
    )
    lc_in_gm_sm = lc_in_gm_sm.drop_vars("time")

    # mask ebro basin
    cds_masked_full = cds_in_gm_sm.where(lc_in_gm_sm.lc_binary == 1)
    cds_masked = mask_dataset_by_geometry(cds_masked_full, gdf.geometry[0])

    soil_moisture_masked_full = soil_moisture.where(lc_in_gm_sm.lc_binary == 1)
    soil_moisture_masked = mask_dataset_by_geometry(
        soil_moisture_masked_full, gdf.geometry[0]
    )

    soil_moisture_chunked = chunk_dataset(
        soil_moisture_masked, chunk_sizes={"time": -1, "lat": 128, "lon": 128}
    )
    cds_chunked = chunk_dataset(cds_masked, chunk_sizes={
        "time": -1, "lat": 128, "lon": 128})

    ds_combined = xr.merge([soil_moisture_chunked, cds_chunked, lc_in_gm_sm])

    chunked_ds = chunk_dataset(
        ds_combined,
        chunk_sizes={"time":-1, "lat":128, "lon":128},
        format_name="zarr",
    )

    store.write_data(chunked_ds, INPUT_FOR_CALIBRATION_ID)

    return f"{INPUT_DIR}/{INPUT_FOR_CALIBRATION_ID}"

