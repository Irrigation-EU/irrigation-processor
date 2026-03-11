from typing import Any

from pydantic import BaseModel


class ChunkSizesConfig(BaseModel):
    time: int = -1
    lat: int = 50
    lon: int = 50

    def to_dict(self) -> dict[str, int]:
        return self.model_dump()


class BaseConfig(BaseModel):
    time_range: list[str]
    bbox: list[float]
    use_gleam: bool


class DataloaderConfig(BaseModel):
    lc_data_id: str
    lc_time_range: list[str]

    cds_data_id: str
    cds_spatial_res: float
    cds_variable_names: list[str]
    cds_optimize_writing: bool

    cds_intermediate_chunks: ChunkSizesConfig = ChunkSizesConfig(
        time=10, lat=178, lon=306
    )
    cds_final_chunks: ChunkSizesConfig = ChunkSizesConfig(time=-1, lat=50, lon=50)
    cds_zappend_spatial_chunks: list[int] = [15, 15]
    clms_zappend_spatial_chunks: list[int] = [150, 150]


class PreprocessingConfig(BaseModel):
    lc_keep_classes: list[int]

    swi_chunks: ChunkSizesConfig = ChunkSizesConfig(time=-1, lat=128, lon=128)
    merged_chunks: ChunkSizesConfig = ChunkSizesConfig(time=-1, lat=50, lon=50)


class CalibrationConfig(BaseModel):
    allowed_months: list[int]
    rainfall_threshold: float
    check_calibration: bool

    calibration_chunks: dict[str, int] = {"params": 4, "lat": 50, "lon": 50}


class SimulationConfig(BaseModel):
    spatial_chunks: ChunkSizesConfig = ChunkSizesConfig(time=1, lat=2072, lon=1708)


class PostprocessingConfig(BaseModel):
    temporal_allowed_months: list[int]
    spatial_mask_zip_url: str
    spatial_mask_filename: str
    spatial_mask_bbox: list[float]
    spatial_mask_threshold: int


class StorageOptionsConfig(BaseModel):
    anon: bool
    key: str
    secret: str
    client_kwargs: dict[str, Any] | None = None


class StoreKwargsConfig(BaseModel):
    root: str
    max_depth: int
    storage_options: StorageOptionsConfig


class StorageConfig(BaseModel):
    store_id: str
    store_kwargs: StoreKwargsConfig


class DaskKwargsConfig(BaseModel):
    n_workers: int
    threads_per_worker: int
    memory_limit: str


class DaskConfig(BaseModel):
    dask_kwargs: DaskKwargsConfig


class AppConfig(BaseModel):
    base: BaseConfig
    dataloader: DataloaderConfig
    preprocessing: PreprocessingConfig
    calibration: CalibrationConfig
    simulation: SimulationConfig = SimulationConfig()
    postprocessing: PostprocessingConfig
    storage: StorageConfig
    dask: DaskConfig
