from pydantic import BaseModel
import xarray as xr

from irrigation_processor.constants import CLMS_DATA_ID, LC_DATA_ID
from src.irrigation_processor.pipeline import FromTask, Step

step = Step()


class BaseContext(BaseModel):
    step_name: str
    time_range: tuple = ("2020-01-01", "2020-01-31")
    bbox: list = [-5, 40, 3, 44]


class DataLoaderContext(BaseContext):
    lc_time: str = "2020-01-01"
    cds_data_id: str = "reanalysis-era5-land"
    cds_variables: tuple = (
        "potential_evaporation",
        "total_precipitation",
    )
    cds_spatial_res: float = 0.1


class PreprocessorContext(BaseContext):
    # TODO: a better way for these paths?
    spatial_mask_path: str = "notebooks/codes/ebro_data/Limite_Cuenca_Ebro84.shp"


class CalibratorContext(BaseContext):
    chunk_size: dict = {"time": -1, "lat": 10, "lon": 10}
    mask_months: list = [5, 6, 7, 8, 9]
    rainfall_threshold: float = 1.0


class SimulatorContext(BaseContext):
    """"""


@step.register(name="dataloader", context_cls=DataLoaderContext)
def dataloader(context: DataLoaderContext):
    from src.irrigation_processor.dataloader import load_data

    return load_data(context)


@step.register(
    inputs=(
        FromTask("dataloader", CLMS_DATA_ID.split(".")[0]),
        FromTask("dataloader", LC_DATA_ID.split(".")[0]),
        FromTask("dataloader", "era5_data_id"),
    ),
    name="preprocessing",
    context_cls=PreprocessorContext,
)
def preprocessing(
    context: PreprocessorContext, sm_cube: xr.Dataset, lc_cube: xr.Dataset, era5_data_id:
        str
):
    from src.irrigation_processor.preprocessor import irrigation_preprocessor
    return irrigation_preprocessor(context, sm_cube, lc_cube, era5_data_id)


@step.register(
    inputs=(FromTask("preprocessing", "preprocessed_path"),),
    name="calibration",
    context_cls=CalibratorContext,
)
def calibration(context: CalibratorContext, preprocessed_path: str):
    from src.irrigation_processor.calibrator import soil_moisture_inversion_calibration

    return soil_moisture_inversion_calibration(context, preprocessed_path)


@step.register(
    inputs=(
        FromTask("preprocessing", "preprocessed_path"),
        FromTask("calibration", "calibrated_path"),
    ),
    name="simulation",
    context_cls=SimulatorContext,
)
def simulation(context: SimulatorContext, preprocessed_path: str, calibrated_path: str):
    from src.irrigation_processor.simulator import irrigation_simulator

    return irrigation_simulator(context, preprocessed_path, calibrated_path)
