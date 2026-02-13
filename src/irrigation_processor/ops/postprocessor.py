import io
import zipfile

import numpy as np
import requests
import rioxarray as rxr
import xarray as xr
from xcube_resampling import resample_in_space
from xcube_resampling.gridmapping import GridMapping

from irrigation_processor.constants import (
    IWU_POSTPROCESSED_ESTIMATES_SPATIAL_ID,
    IWU_POSTPROCESSED_ESTIMATES_TEMPORAL_ID)
from irrigation_processor.utils import get_existing_data


def postprocessor(context, iwu_spatial_path: str, iwu_temporal_path: str, dask_client):
    store = context.store

    result_spatial = get_existing_data(
        store=store,
        data_id=IWU_POSTPROCESSED_ESTIMATES_SPATIAL_ID,
    )
    result_temporal = get_existing_data(
        store=store,
        data_id=IWU_POSTPROCESSED_ESTIMATES_TEMPORAL_ID,
    )
    if result_spatial is not None and result_temporal is not None:
        return {
            "iwu_postprocessed_spatial_path": result_spatial,
            "iwu_postprocessed_temporal_path": result_temporal,
        }

    iwu_spatial = store.open_data(iwu_spatial_path)
    iwu_temporal = store.open_data(iwu_temporal_path)

    assert np.all(iwu_spatial.time.diff("time").values.astype("timedelta64[D]") == 14)
    assert np.all(iwu_temporal.time.diff("time").values.astype("timedelta64[D]") == 14)

    iwu_spatial_masked, iwu_temporal_masked = _do_temporal_masking(
        context, iwu_spatial, iwu_temporal
    )
    filtered_spatial, filtered_temporal = _do_spatial_masking(
        context, iwu_spatial_masked, iwu_temporal_masked
    )

    assert np.all(
        filtered_spatial.time.diff("time").values.astype("timedelta64[D]") == 14
    )
    store.write_data(filtered_spatial,
                     IWU_POSTPROCESSED_ESTIMATES_SPATIAL_ID, replace=False)

    assert np.all(
        filtered_temporal.time.diff("time").values.astype("timedelta64[D]") == 14
    )
    store.write_data(filtered_temporal, IWU_POSTPROCESSED_ESTIMATES_TEMPORAL_ID, replace=False)

    return {
        "iwu_postprocessed_spatial_path": IWU_POSTPROCESSED_ESTIMATES_SPATIAL_ID,
        "iwu_postprocessed_temporal_path": IWU_POSTPROCESSED_ESTIMATES_TEMPORAL_ID,
    }


def _get_spatial_mask(context) -> xr.Dataset:
    url: str = context.spatial_mask_zip_url
    spatial_mask_filename: str = context.spatial_mask_filename
    r = requests.get(url)
    z = zipfile.ZipFile(io.BytesIO(r.content))

    with z.open(spatial_mask_filename) as asc:
        da = rxr.open_rasterio(asc)
    da = da.drop_vars("band", errors="ignore")
    return da.to_dataset(name="band_1")


def _do_temporal_masking(
    context,
    iwu_spatial: xr.Dataset,
    iwu_temporal: xr.Dataset,
):
    temporal_allowed_months: list[int] = context.temporal_allowed_months
    iwu_spatial_masked = iwu_spatial.where(
        iwu_spatial.time.dt.month.isin(temporal_allowed_months), 0
    )
    iwu_temporal_masked = iwu_temporal.where(
        iwu_temporal.time.dt.month.isin(temporal_allowed_months), 0
    )

    return iwu_spatial_masked, iwu_temporal_masked


def _do_spatial_masking(
    context,
    iwu_spatial: xr.Dataset,
    iwu_temporal: xr.Dataset,
):
    bbox: list[float] = context.spatial_mask_bbox
    spatial_mask = _get_spatial_mask(context)
    threshold: int = context.spatial_mask_threshold
    spatial_mask_subset = spatial_mask.sel(
        y=slice(bbox[3], bbox[1]), x=slice(bbox[0], bbox[2])
    )
    spatial_mask_subset = spatial_mask_subset.rename({"x": "lon", "y": "lat"})

    gm_is = GridMapping.from_dataset(iwu_spatial)
    gm_it = GridMapping.from_dataset(iwu_temporal)

    ds_in_gm_is = resample_in_space(
        spatial_mask_subset, target_gm=gm_is, interp_methods=1
    )
    ds_in_gm_is = ds_in_gm_is.rename({"band_1": "mask"})

    ds_in_gm_it = resample_in_space(
        spatial_mask_subset, target_gm=gm_it, interp_methods=1
    )
    ds_in_gm_it = ds_in_gm_it.rename({"band_1": "mask"})
    filtered_spatial = (
        iwu_spatial["iwu_est"].where(ds_in_gm_is["mask"] > threshold).squeeze()
    )
    filtered_spatial = filtered_spatial.to_dataset(name="iwu_est")
    filtered_temporal = (
        iwu_temporal["iwu_est"].where(ds_in_gm_it["mask"] > threshold).squeeze()
    )
    filtered_temporal = filtered_temporal.to_dataset(name="iwu_est")
    return filtered_spatial, filtered_temporal
