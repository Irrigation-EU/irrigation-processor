from src.irrigation_processor.pipeline import FromTask, Step

from pydantic import BaseModel


step = Step()

class BaseContext(BaseModel):
    step_name: str
    time_period: tuple = ("2020-01-01", "2020-01-31")

@step.register(
    inputs=(FromTask("preprocessing", "preprocessed"),),
    outputs=("calibrated",),
    name="calibration",
    context_cls=BaseContext
)
def calibration(context, preprocessed):
    print("inside calibration, got preprocessed:", preprocessed)
    return {"calibrated": {"data": [x * 2 for x in preprocessed["data"]]}}


@step.register(
    outputs=("preprocessed",),
    name="preprocessing",
    context_cls=BaseContext
)
def preprocessing(context: BaseContext):
    print("inside preprocessing, context:", context.time_period)
    return {"preprocessed": {"data": [1, 2, 3]}, "int_val": 44, "str_val": "meow"}

@step.register(
    inputs=(FromTask("calibration", "calibrated"),),
    outputs=("model",),
    name="training",
    context_cls=BaseContext
)
def training(context, calibrated):
    print("inside training, got calibrated:", calibrated)
    return {"model": {"coef": sum(calibrated["data"])}}

