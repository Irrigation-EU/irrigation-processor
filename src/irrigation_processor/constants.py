import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("irrigation-processor")

# Data ids
CLMS_DATA_ID = "soil_moisture.zarr"
ERA5_DATA_ID = "era5.zarr"
LC_DATA_ID = "landcover2020global.zarr"
PROCESSED_CLMS_DATA_ID = "soil_moisture_filled.zarr"
INPUT_FOR_CALIBRATION_ID = "irrigation_input.zarr"

# xcube file data store name
INPUT_DIR = "input_irrigation"