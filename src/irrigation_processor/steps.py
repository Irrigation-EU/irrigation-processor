from src.irrigation_processor.pipeline import FromTask, step

class BaseContext:
    def __init__(self, **kwargs):
        self._meta = kwargs
    def get_meta(self):
        return self._meta

@step(inputs=(FromTask("preprocessing", "preprocessed"),), outputs=("calibrated",), name="calibration", context_cls=BaseContext)
def calibration(context, preprocessed):
    print("inside calibration, got preprocessed:", preprocessed)
    return {"calibrated": {"data": [x * 2 for x in preprocessed["data"]]}}

@step(outputs=("preprocessed",), name="preprocessing", context_cls=BaseContext)
def preprocessing(context):
    print("inside preprocessing, context:", context.get_meta())
    return {"preprocessed": {"data": [1, 2, 3]}, "int_val": 44, "str_val": "meow"}

@step(inputs=(FromTask("calibration", "calibrated"),), outputs=("model",), name="training", context_cls=BaseContext)
def training(context, calibrated):
    print("inside training, got calibrated:", calibrated)
    return {"model": {"coef": sum(calibrated["data"])}}