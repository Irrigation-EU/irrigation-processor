import numpy as np
import xarray as xr
from scipy.optimize import minimize
from xcube.core.chunk import chunk_dataset
from xcube.core.store import new_data_store

from irrigation_processor.constants import (
    CALIBRATED_PARAMS_ID,
    INPUT_DIR,
    INPUT_FOR_CALIBRATION_ID,
    logger,
)
from irrigation_processor.steps import CalibratorContext

store = new_data_store("file", root=INPUT_DIR)


def soil_moisture_inversion_calibration(
    context: CalibratorContext, input_path: str
) -> dict:
    logger.info(f"calibrating... {context} {input_path}")
    irr_input = store.open_data(INPUT_FOR_CALIBRATION_ID)
    chunked_irr_input = chunk_dataset(irr_input, context.chunk_size)

    tp = chunked_irr_input["tp"]
    mask_season = tp["time"].dt.month.isin(context.mask_months)
    masked_tp = tp.where(~(mask_season & (tp < context.rainfall_threshold)))

    result = xr.apply_ufunc(
        calib_wrapper,
        chunked_irr_input["SWI"],
        masked_tp,
        chunked_irr_input["pev"],
        7,
        input_core_dims=[["time"], ["time"], ["time"], []],
        output_core_dims=[["params"]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float],
        output_sizes={"params": 4},
    )

    result = result.assign_coords(params=["a", "b", "z", "RF"])

    # TODO: Dask setup?

    ds_calib = result.compute()

    store.write_data(ds_calib.to_dataset(name="calibration"), CALIBRATED_PARAMS_ID)

    calibrated_path = f"{INPUT_DIR}/{CALIBRATED_PARAMS_ID}"
    logger.info(f"calibration complete...{calibrated_path}")
    return {"calibrated_path": calibrated_path}


def sm_inversion(sm, et, a, b, z, RF, thr=None):
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
    sm, p_obs, et, NN, x0=None, bounds=None, options=None, method="TNC"
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


def cost_fun(x0, sm, p_obs, et, NN):
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


def calib_wrapper(sm_ts, p_obs_ts, et_ts, NN):
    if np.isnan(np.nanmean(sm_ts)):
        return np.array([np.nan, np.nan, np.nan, np.nan])

    a, b, z, RF = calib_sm_inversion(sm_ts, p_obs_ts, et_ts, NN)
    return np.array([a, b, z, RF])
