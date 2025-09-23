from pydantic import BaseModel

from src.irrigation_processor.pipeline import FromTask, Step

step = Step()


class BaseContext(BaseModel):
    step_name: str
    time_range: tuple = ("2020-01-01", "2020-01-31")
    bbox: list = []
    crs: str = ""


class DataLoaderContext(BaseContext):
    """"""


class PreprocessorContext(BaseContext):
    """"""


class CalibratorContext(BaseContext):
    """"""


class SimulatorContext(BaseContext):
    """"""


@step.register(name="dataloader", context_cls=BaseContext)
def dataloader(context: DataLoaderContext):
    from src.irrigation_processor.dataloader import load_data

    return load_data(context)


@step.register(
    inputs=(
        FromTask("dataloader", "sm_path"),
        FromTask("dataloader", "lc_path"),
        FromTask("dataloader", "era5_path"),
    ),
    name="preprocessing",
    context_cls=PreprocessorContext,
)
def preprocessing(
    context: PreprocessorContext, sm_path: str, lc_path: str, era5_path: str
):
    from src.irrigation_processor.preprocessor import irrigation_preprocessor

    return irrigation_preprocessor(context, sm_path, lc_path, era5_path)


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
