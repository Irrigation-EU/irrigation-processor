import numpy as np
import xarray as xr
from xcube.core.chunk import chunk_dataset

from irrigation_processor.config import AppConfig
from irrigation_processor.constants import (IWU_ESTIMATES_SPATIAL_ID,
                                            IWU_ESTIMATES_TEMPORAL_ID, LOG)
from irrigation_processor.core.storage import Storage
from irrigation_processor.utils import get_existing_data, validate_dataset


def irrigation_simulator(
    context: AppConfig,
    storage: Storage,
    preprocessed_ds: xr.Dataset,
    calibrated_path: str,
) -> dict:
    LOG.info("simulating irrigation water use...")

    result_spatial = get_existing_data(
        storage=storage,
        data_id=IWU_ESTIMATES_SPATIAL_ID,
    )
    result_temporal = get_existing_data(
        storage=storage,
        data_id=IWU_ESTIMATES_TEMPORAL_ID,
    )
    if result_spatial is not None and result_temporal is not None:
        return {
            "iwu_spatial_estimates": result_spatial,
            "iwu_temporal_estimates": result_temporal,
        }

    calibration = storage.load(calibrated_path)

    validate_dataset(context, preprocessed_ds)

    assert calibration.sizes["params"] == 4, (
        f"4 params expected, got {calibration.sizes['params']}"
    )

    psim = xr.apply_ufunc(
        _ts_smet4irr,
        preprocessed_ds["SWI"],
        preprocessed_ds["pev"],
        calibration["calibration"].sel(params="a"),
        calibration["calibration"].sel(params="b"),
        calibration["calibration"].sel(params="z"),
        calibration["calibration"].sel(params="RF"),
        input_core_dims=[["time"], ["time"], [], [], [], []],
        output_core_dims=[["time2"]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float],
        dask_gufunc_kwargs={
            "output_sizes": {"time2": preprocessed_ds.sizes["time"] - 1}
        },
    )

    psim2 = psim.rename({"time2": "time"})
    time_coord = preprocessed_ds["time"].values[:-1]
    psim2 = psim2.assign_coords(time=time_coord)
    psim2 = psim2.transpose("time", "lat", "lon")

    tp_except_last_timestamp = preprocessed_ds["tp"].isel(time=slice(0, -1))

    tp_weekly = _resample_sum(tp_except_last_timestamp, step=7)  # 7 days
    psim2_weekly = _resample_sum(psim2, step=7)  # 7 days

    IRR = psim2_weekly - tp_weekly
    IRR_clipped = IRR.clip(0, 1000)
    IRR_weekly = IRR_clipped.where(IRR_clipped / tp_weekly >= 0.2, 0)

    IRR_biweekly = _resample_sum(IRR_weekly, step=2)  # 14 days

    assert np.all(
        IRR_biweekly.to_dataset(name="iwu_est")
        .time.diff("time")
        .values.astype("timedelta64[D]")
        == 14
    )

    storage.save(
        IWU_ESTIMATES_TEMPORAL_ID,
        IRR_biweekly.to_dataset(name="iwu_est"),
    )

    IRR_biweekly_temporal = storage.load(IWU_ESTIMATES_TEMPORAL_ID)
    IRR_biweekly_spatial = chunk_dataset(
        IRR_biweekly_temporal,
        context.simulation.spatial_chunks.to_dict(),
        format_name="zarr",
    )

    assert np.all(
        IRR_biweekly_temporal.time.diff("time").values.astype("timedelta64[D]") == 14
    )

    storage.save(IWU_ESTIMATES_SPATIAL_ID, IRR_biweekly_spatial)

    LOG.info("simulation complete...")
    return {
        "iwu_spatial_estimates": IWU_ESTIMATES_SPATIAL_ID,
        "iwu_temporal_estimates": IWU_ESTIMATES_TEMPORAL_ID,
    }


def _ts_smet4irr(
    sm: np.ndarray,
    et: np.ndarray,
    a: float,
    b: float,
    z: float,
    RF: float,
    thr: float | None = None,
):
    p_sim = (
        z * (sm[1:] - sm[:-1])
        + ((a * sm[1:] ** b + a * sm[:-1] ** b) / 2.0)
        + ((RF * sm[1:] * et[1:] + RF * sm[:-1] * et[:-1]) / 2.0)
    )

    p_sim[abs(np.diff(sm)) <= 0.001] = 0.0
    p_sim[p_sim < 1.0] = 0.0
    return np.clip(p_sim, 0, thr)


def _resample_sum(dataarray: xr.DataArray, step: int) -> xr.DataArray:
    data = dataarray.values
    steps = dataarray.time.size // step
    data = data[: steps * step, :, :]
    data = data.reshape(steps, step, dataarray.sizes["lat"], dataarray.sizes["lon"])
    data_sum = data.sum(axis=1)
    return xr.DataArray(
        data=data_sum,
        dims=("time", "lat", "lon"),
        coords=dict(
            time=dataarray.time[::step][:steps],
            lat=dataarray.lat,
            lon=dataarray.lon,
            spatial_ref=dataarray.spatial_ref,
        ),
        attrs=dataarray.attrs,
    )
