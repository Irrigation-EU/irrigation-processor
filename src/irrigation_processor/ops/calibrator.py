import numpy as np
import xarray as xr
from pydantic import BaseModel
from scipy.optimize import minimize
from xcube.core.chunk import chunk_dataset

from irrigation_processor.constants import CALIBRATED_ID, LOG, OUTPUT_DIR
from irrigation_processor.utils import get_existing_data


def soil_moisture_inversion_calibration(
    context: BaseModel, preprocessed_data: xr.Dataset, dask_client
) -> dict:
    store = context.store

    result = get_existing_data(
        store=store,
        data_id=CALIBRATED_ID,
    )
    if result is not None:
        if context.check_calibration:
            calibrated = store.open_data("calibrated.zarr")
            arr_reshaped = calibrated.calibration.values.reshape(-1, 4)
            unique_param_sets = np.unique(arr_reshaped, axis=0)

            LOG.info(f"Number of unique parameter sets: {len(unique_param_sets)}")
            unique_param_sets_no_nan = unique_param_sets[
                ~np.isnan(unique_param_sets).any(axis=1)
            ]
            LOG.info(
                f"Unique parameter sets without NaNs: {len(unique_param_sets_no_nan)}"
            )
        return {"calibrated_data_id": result}

    LOG.info(f"calibrating... {context} {preprocessed_data}")
    tp = preprocessed_data["tp"]
    mask_season = tp["time"].dt.month.isin(context.allowed_months)
    masked_tp = tp.where(~(mask_season & (tp < context.rainfall_threshold)))
    LOG.info("data masked")
    result = xr.apply_ufunc(
        calib_wrapper,
        preprocessed_data["SWI"],
        masked_tp,
        preprocessed_data["pev"],
        7,
        input_core_dims=[["time"], ["time"], ["time"], []],
        output_core_dims=[["params"]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float],
        output_sizes={"params": 4},
    )

    result = result.assign_coords(params=["a", "b", "z", "RF"])

    result = result.to_dataset(name="calibration")
    LOG.info("dask: data calibration beginnning")

    subresults = []
    step = 200
    for i in range(0, result.sizes["lat"], step):
        subresults.append(result.isel(lat=slice(i, i + step)))

    for i, subresult in enumerate(subresults):
        if store.has_data(f"calibrated_{i}.zarr"):
            continue
        store.write_data(subresult, f"calibrated_{i}.zarr", replace=True)

    data_ids = store.list_data_ids()
    data_ids_cal = [data_id for data_id in data_ids if "calibrated_" in data_id]

    datasets = []
    for data_id in sorted(data_ids_cal):
        datasets.append(store.open_data(data_id))
    ds = xr.concat(datasets, dim="lat", join="left")
    chunked_ds = chunk_dataset(
        ds,
        chunk_sizes={"params": 4, "lat": 50, "lon": 50},
        format_name="zarr",
    )
    store.write_data(chunked_ds, CALIBRATED_ID, replace=True)
    for data_id in data_ids_cal:
        store.delete_data(data_id)
    calibrated_path = f"{OUTPUT_DIR}/{CALIBRATED_ID}"
    LOG.info(f"calibration complete...{calibrated_path}")
    if context.check_calibration:
        calibrated = store.open_data("calibrated.zarr")
        arr_reshaped = calibrated.calibration.values.reshape(-1, 4)
        unique_param_sets = np.unique(arr_reshaped, axis=0)

        LOG.info(f"Number of unique parameter sets: {len(unique_param_sets)}")
        unique_param_sets_no_nan = unique_param_sets[
            ~np.isnan(unique_param_sets).any(axis=1)
        ]
        LOG.info(f"Unique parameter sets without NaNs: {len(unique_param_sets_no_nan)}")
    return {"calibrated_data_id": CALIBRATED_ID}


def sm_inversion(
    sm: np.ndarray,
    et: np.ndarray,
    a: float,
    b: float,
    z: float,
    RF: float,
    thr: float | None = None,
):
    """Evotranspiration and Soil moisture to irrigation"""
    # sm - soil moisture
    # et - evotranspiration
    # a/b - drainage parameter
    # z - soil water capacity
    # f/RF - Adjusting factor
    p_sim = (
        z * (sm[1:] - sm[:-1])
        + ((a * sm[1:] ** b + a * sm[:-1] ** b) / 2.0)
        + ((RF * sm[1:] * et[1:] + RF * sm[:-1] * et[:-1]) / 2.0)
    )

    p_sim[abs(np.diff(sm)) <= 0.001] = 0.0
    p_sim[p_sim < 1.0] = 0.0  # 2mm/day for NN=4 -> 0.5

    return np.clip(p_sim, 0, thr)


def calib_sm_inversion(
    sm: np.ndarray,
    p_obs: np.ndarray,
    et: np.ndarray,
    NN: int,
    x0: np.ndarray | None = None,
    bounds: tuple | None = None,
    options: dict | None = None,
    method: str = "TNC",
):
    if x0 is None:
        x0 = np.array([20.0, 5.0, 80, 1.0])

    if bounds is None:
        bounds = ((0, 200), (0.01, 50), (1, 800), (0.1, 1.4))

    if options is None:
        options = {"ftol": 1e-8, "maxfun": 4000, "disp": False}

    result = minimize(
        cost_fun,
        x0,
        args=(sm, p_obs, et, NN),
        method=method,
        bounds=bounds,
        options=options,
    )

    a, b, z, RF = result.x

    return a, b, z, RF


def cost_fun(
    x0: np.ndarray, sm: np.ndarray, p_obs: np.ndarray, et: np.ndarray, NN: int
):
    # The following args are 1D time-series
    p_sim = sm_inversion(sm, et, x0[0], x0[1], x0[2], x0[3])
    p_obs = p_obs[:-1]
    p_sim[np.isnan(p_obs)] = np.nan
    p_sim1 = np.add.reduceat(
        p_sim[np.isfinite(p_sim)], np.arange(0, len(p_sim[np.isfinite(p_sim)]), NN)
    )
    p_obs1 = np.add.reduceat(
        p_obs[np.isfinite(p_sim)], np.arange(0, len(p_obs[np.isfinite(p_sim)]), NN)
    )
    rmsd = np.nanmean((p_obs1 - p_sim1) ** 2) ** 0.5

    return rmsd


def calib_wrapper(sm_ts: np.ndarray, p_obs_ts: np.ndarray, et_ts: np.ndarray, NN: int):
    if np.isnan(np.nanmean(sm_ts)):
        return np.array([np.nan, np.nan, np.nan, np.nan])

    a, b, z, RF = calib_sm_inversion(sm_ts, p_obs_ts, et_ts, NN)
    return np.array([a, b, z, RF])
