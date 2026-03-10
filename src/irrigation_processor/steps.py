import xarray as xr
from dask.distributed import Client

from irrigation_processor.config import AppConfig
from irrigation_processor.constants import INPUT_FOR_CALIBRATION_ID
from irrigation_processor.core.pipeline import FromStep, StepRegistry
from irrigation_processor.core.storage import Storage

registry = StepRegistry()


@registry.step(
    name="dataloader",
    outputs=("sm_data_id", "lc_data_id", "era5_vars_data_id", "gleam_data_id"),
)
def dataloader(context: AppConfig, storage: Storage):
    from irrigation_processor.ops.dataloader import load_data

    return load_data(context, storage)


@registry.step(
    name="preprocessing",
    inputs=(
        FromStep("dataloader", "sm_data_id"),
        FromStep("dataloader", "lc_data_id"),
        FromStep("dataloader", "era5_vars_data_id"),
        FromStep("dataloader", "gleam_data_id"),
    ),
    outputs=(INPUT_FOR_CALIBRATION_ID,),
)
def preprocessing(
    context: AppConfig,
    storage: Storage,
    sm_data_id: str,
    lc_data_id: str,
    era5_vars_data_id: str,
    gleam_data_id: str,
):
    from irrigation_processor.ops.preprocessor import irrigation_preprocessor

    return irrigation_preprocessor(
        context,
        storage,
        sm_data_id,
        lc_data_id,
        era5_vars_data_id,
        gleam_data_id,
    )


@registry.step(
    name="calibration",
    inputs=(FromStep("preprocessing", INPUT_FOR_CALIBRATION_ID),),
    outputs=("calibrated_data_id",),
)
def calibration(
    context: AppConfig,
    storage: Storage,
    dask_client: Client,
    preprocessed_data: xr.Dataset,
):
    from irrigation_processor.ops.calibrator import \
        soil_moisture_inversion_calibration

    return soil_moisture_inversion_calibration(
        context, storage, dask_client, preprocessed_data
    )


@registry.step(
    name="simulation",
    inputs=(
        FromStep("preprocessing", INPUT_FOR_CALIBRATION_ID),
        FromStep("calibration", "calibrated_data_id"),
    ),
    outputs=("iwu_spatial_estimates", "iwu_temporal_estimates"),
)
def simulation(
    context: AppConfig,
    storage: Storage,
    preprocessed_path: xr.Dataset,
    calibrated_path: str,
):
    from irrigation_processor.ops.simulator import irrigation_simulator

    return irrigation_simulator(context, storage, preprocessed_path, calibrated_path)


@registry.step(
    name="postprocessing",
    inputs=(
        FromStep("simulation", "iwu_spatial_estimates"),
        FromStep("simulation", "iwu_temporal_estimates"),
    ),
    outputs=("iwu_postprocessed_spatial_path", "iwu_postprocessed_temporal_path"),
)
def postprocessing(
    context: AppConfig,
    storage: Storage,
    iwu_spatial_path: str,
    iwu_temporal_path: str,
):
    from irrigation_processor.ops.postprocessor import postprocessor

    return postprocessor(context, storage, iwu_spatial_path, iwu_temporal_path)
