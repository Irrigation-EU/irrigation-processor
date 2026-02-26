# Changes in version 0.1.0 (in development)

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