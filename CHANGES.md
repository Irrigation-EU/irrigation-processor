# Changes in version 0.1.1

- Added documentation for key classes.
- Step functions can now optionally receive the Dask `Client` by including 
  `dask_client` in the arguments (e.g., `your_step(config, storage, dask_client, *your_args)`).
- Improved calibration performance by using chunked processing by setting `vectorize=False` 
  in `xr.apply_ufunc`. and restarting the Dask cluster after each written batch. 
- Updated `README.md` and test suite.
- Updated support for the GLEAM dataset to use its `Potential Evaporation`
  variable instead of ERA5-Land.
# Changes in version 0.1.0

- Implemented `Pipeline` framework that allows the user to 
  - create pipelines easily using steps
  - allows creating DAGs automatically by inferring the dependencies
  - handles the intermediate storage between the steps automatically
  - logs the outputs of each step to a file
- Implemented initial `irrigation-processor` pipeline containing 5 steps
  - dataloader
  - preprocessing
  - calibration
  - simulation
  - postprocessing
- Added configurable chunk sizes for all pipeline steps via `config.yml`
- Standardized the test suite using centralized helpers.
- Integrated `typer` for running the processor with CLI.
- Improved Pydantic configuration models for better type safety and validation
- Integrated `pixi` for standardized environment management
- Added initial documentation