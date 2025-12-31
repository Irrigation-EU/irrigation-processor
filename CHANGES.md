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
- Add tests and initial documentation