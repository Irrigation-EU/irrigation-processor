import xarray as xr
from pydantic import BaseModel

from irrigation_processor.constants import INPUT_FOR_CALIBRATION_ID, LC_DATA_ID
from irrigation_processor.core.pipeline import FromStep, StepRegistry

registry = StepRegistry()


@registry.step(name="dataloader", outputs=("sm_data_id", LC_DATA_ID, "era5_data_id"))
def dataloader(context: BaseModel):
    from irrigation_processor.ops.dataloader import load_data

    return load_data(context)


@registry.step(
    name="preprocessing",
    inputs=(
        FromStep("dataloader", "sm_data_id"),
        FromStep("dataloader", LC_DATA_ID),
        FromStep("dataloader", "era5_data_id"),
    ),
    outputs=(INPUT_FOR_CALIBRATION_ID,),
)
def preprocessing(
    context: BaseModel, sm_data_id: str, lc_cube: xr.Dataset, era5_data_id: str
):
    from irrigation_processor.ops.preprocessor import irrigation_preprocessor

    return irrigation_preprocessor(context, sm_data_id, lc_cube, era5_data_id)


# This framework also provides the capability to use dask in specific tasks
# as required. Just pass in the dask_client as the last argument as shown
# below and propagate it to your function and use it there.
@registry.step(
    name="calibration",
    inputs=(FromStep("preprocessing", INPUT_FOR_CALIBRATION_ID),),
    outputs=("calibrated_data_id",),
)
def calibration(context: BaseModel, preprocessed_data: xr.Dataset, dask_client):
    from irrigation_processor.ops.calibrator import soil_moisture_inversion_calibration

    return soil_moisture_inversion_calibration(context, preprocessed_data, dask_client)


@registry.step(
    name="simulation",
    inputs=(
        FromStep("preprocessing", INPUT_FOR_CALIBRATION_ID),
        FromStep("calibration", "calibrated_data_id"),
    ),
    outputs=("iwu_spatial_estimates", "iwu_temporal_estimates"),
)
def simulation(
    context: BaseModel, preprocessed_path: xr.Dataset, calibrated_path: str, dask_client
):
    from irrigation_processor.ops.simulator import irrigation_simulator

    return irrigation_simulator(
        context, preprocessed_path, calibrated_path, dask_client
    )


@registry.step(
    name="postprocessing",
    inputs=(
        FromStep("simulation", "iwu_spatial_estimates"),
        FromStep("simulation", "iwu_temporal_estimates"),
    ),
    outputs=(),
)
def postprocessing(
    context: BaseModel, iwu_spatial_path: str, iwu_temporal_path: str, dask_client
):
    from irrigation_processor.ops.postprocessor import postprocessor

    return postprocessor(context, iwu_spatial_path, iwu_temporal_path, dask_client)
