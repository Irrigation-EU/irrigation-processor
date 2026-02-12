import logging
import sys

logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger("irrigation-processor")
handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s")
handler.setFormatter(formatter)

LOG.addHandler(handler)
LOG.propagate = False

# Data ids
CLMS_DATA_ID = "soil_moisture.zarr"
ERA5_DATA_ID = "era5.zarr"
LC_DATA_ID = "landcover2020Europe.zarr"
GLEAM_DATA_ID = "gleamv4_2.zarr"
PROCESSED_CLMS_DATA_ID = "soil_moisture_filled.zarr"
INPUT_FOR_CALIBRATION_ID = "irrigation_input.zarr"
CALIBRATED_ID = "calibrated.zarr"
IWU_ESTIMATES_SPATIAL_ID = "iwu_estimates_spatial.zarr"
IWU_ESTIMATES_TEMPORAL_ID = "iwu_estimates_temporal.zarr"
IWU_POSTPROCESSED_ESTIMATES_SPATIAL_ID = "iwu_postprocessed_spatial.zarr"
IWU_POSTPROCESSED_ESTIMATES_TEMPORAL_ID = "iwu_postprocessed_temporal.zarr"

# Pipeline results
PIPELINE_RESULTS_CACHE_DIR = ".pipeline_results_cache"
