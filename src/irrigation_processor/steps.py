from pydantic import BaseModel
import xarray as xr

from irrigation_processor.constants import CLMS_DATA_ID, LC_DATA_ID
from src.irrigation_processor.pipeline import FromTask, Step

step = Step()

@step.register(name="dataloader")
def dataloader(context: BaseModel):
    from src.irrigation_processor.dataloader import load_data
    return load_data(context)


@step.register(
    inputs=(
        FromTask("dataloader", CLMS_DATA_ID),
        FromTask("dataloader", LC_DATA_ID),
        FromTask("dataloader", "era5_data_id"),
    ),
    outputs=(
        "preprocessed.zarr",
    ),
    name="preprocessing",
)
def preprocessing(
    context: BaseModel, sm_cube: xr.Dataset, lc_cube: xr.Dataset, era5_data_id:
        str
):
    from src.irrigation_processor.preprocessor import irrigation_preprocessor
    return irrigation_preprocessor(context, sm_cube, lc_cube, era5_data_id)


@step.register(
    inputs=(FromTask("preprocessing", "preprocessed_path"),),
    name="calibration",
)
def calibration(context: BaseModel, preprocessed_path: str):
    from src.irrigation_processor.calibrator import soil_moisture_inversion_calibration

    return soil_moisture_inversion_calibration(context, preprocessed_path)


@step.register(
    inputs=(
        FromTask("preprocessing", "preprocessed_path"),
        FromTask("calibration", "calibrated_path"),
    ),
    name="simulation",
)
def simulation(context: BaseModel, preprocessed_path: str, calibrated_path: str):
    from src.irrigation_processor.simulator import irrigation_simulator

    return irrigation_simulator(context, preprocessed_path, calibrated_path)
