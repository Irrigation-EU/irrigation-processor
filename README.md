# 🌱 Irrigation Processor

[![CI](https://github.com/Irrigation-EU/irrigation-processor/actions/workflows/ci.yml/badge.svg)](https://github.com/Irrigation-EU/irrigation-processor/actions/workflows/ci.yml)
[![codecov](https://codecov.io/github/Irrigation-EU/irrigation-processor/graph/badge.svg?token=3KFCF7C6DA)](https://codecov.io/github/Irrigation-EU/irrigation-processor)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v0.json)](https://github.com/charliermarsh/ruff)
[![isort](https://img.shields.io/badge/imports-isort-1674b1.svg)](https://pycqa.github.io/isort/)
[![Pixi](https://img.shields.io/badge/env-pixi-5A67D8.svg)](https://pixi.sh)


Irrigation Processor is a scientific Python package for estimating irrigation 
water use (IWU) across **Europe** from soil moisture, meteorological variables, and land-cover 
data using a reproducible, multi-step processing pipeline.

The package is designed for large-scale geospatial analysis, supports caching 
and restartability.

This package was developed for the Irrigation-EU project funded by ESA.

## Overview

The package implements a five-step pipeline:

1. Data loading
2. Preprocessing
3. Calibration
4. Simulation
5. Postprocessing

Each step:
- consumes outputs from previous steps
- writes its results to a storage
- can be skipped automatically if outputs already exist

## Credentials needed:

### CLMS API creds

To get these creds, please follow the steps [here](https://eea.github.io/clms-api-docs/authentication.html)

### CDSE AWS creds

To get these creds, please follow the steps [here](https://documentation.dataspace.copernicus.eu/APIs/S3.html#generate-secrets)

Once you have your secrets, before running the pipeline, run the following commands

```bash
export CDSE_AWS_ACCESS_KEY_ID=<your key>
export CDSE_AWS_SECRET_ACCESS_KEY=<your secret>
export CDSE_AWS_ENDPOINT_URL=https://eodata.dataspace.copernicus.eu/
```

or add them to `.env` file as shown below:

```.dotenv
# The following are for CDSE S3 access to CLMS data
CDSE_AWS_ACCESS_KEY_ID=
CDSE_AWS_SECRET_ACCESS_KEY=
CDSE_AWS_ENDPOINT_URL=
```

## Configuration file (`config.yml`)

The pipeline is configured using a YAML file called `config.yml` stored on the 
root folder.

The configuration file is structured around step names, which correspond 
directly to the steps registered in the pipeline.

**Example basic structure**

```yaml
base:
  # shared configuration for all steps
  time_range:
    - "2020-01-01"
    - "2020-12-31"

calibration:
  # step-specific configuration and additions to base config
  check_calibration: true
  rainfall_threshold: 0.1

postprocessing:
  temporal_allowed_months: [4, 5, 6, 7, 8, 9]
  spatial_mask_threshold: 0.5
```

(keep in mind, the configuration file above is an example)

The `base` section contains shared configuration that is automatically merged 
into every step.

### How step configuration is mapped

Each top-level key in `config.yml` (except `base`) must match a 
registered step name.

For example:

```yaml
calibration:
  check_calibration: true
```
maps directly to the step defined in `steps.py` as:

```python
@registry.step(name="calibration", ...)
def calibration(context, ...):
    from your_pipeline import calibration
    calibration(context, ...)
```

Now, the configurations are available to your function via the 
`context` parameter.

To access any config variable in your function, do:

```python
check_calibration = context.check_calibration
```

If a config key does not correspond to a registered step, the pipeline will raise an error.

## How the pipeline is defined (`StepRegistry`)

All pipeline steps are registered using a shared `StepRegistry`.

```python
registry = StepRegistry()
```

Each step is registered using the `@registry.step` decorator.

This registration:

- records the step name
- records its inputs and outputs
- allows the pipeline to build a dependency graph
- allows dynamic configuration injection

### The `@registry.step` decorator

A pipeline step is defined as a normal Python function, decorated with 
`@registry.step`.

Example:
```python
@registry.step(
    name="preprocessing",
    inputs=(
        FromStep("dataloader", "sm_data_id"),
        FromStep("dataloader", "lc_data_id"),
    ),
    outputs=("preprocessed_data",),
)
def preprocessing(context, sm_data_id, lc_data):
    ...
```

**Parameters explained:**

`name`

The unique name of the step.
This name:

- must be unique
- must match the config file section name
- is used to build dependencies

`inputs`

Defines where this step gets its inputs from.

Inputs are declared using `FromStep(step_name, output_key)`.

This means:

- take the output named "sm_data_id"
- produced by the "dataloader" step
- and pass it as a function argument

The order of inputs must match the function signature.

`outputs`

Declares what this step produces.

Example:

```python
outputs=("iwu_spatial_estimates", "iwu_temporal_estimates")
```

The above outputs declaration shows that this step returns two objects
either as a dict or tuple/list which are then named as per the returned order.

This means
- the step must either return a dict, tuple/list or single output
- the outputs must be defined here in the same order as they are returned by 
the step.
- these outputs are available to the downstream steps only.

How `FromStep` works

`FromStep` defines explicit data dependencies between steps.

```python
FromStep("simulation", "iwu_spatial_estimates")
```

This tells the pipeline:
- the current step depends on simulation
- specifically on its `iwu_spatial_estimates` output
- the pipeline must execute simulation first

Internally, this is used to:

- build a dependency graph (DAG)
- determine execution order
- wire outputs to inputs automatically

One must make sure that the inputs and outputs are ordered correctly as per 
their step function definition.

`Context injection`

Each step receives a `context` object as its first argument.

```python
def calibration(context, preprocessed_data):
    ...
```

The context:

- is a Pydantic model created dynamically
- contains the merged config (base + step config)
- provides access to the shared data store (context.store) using which one can 
check what data already exists and write to it.

`dask_client`

Each step can also optionally recieve the `dask_client` if they wish to run their
step using the local dask cluster to fasten the computations a bit.

To get this `dask_client`, add it as the last argument to your step

For example:

```python
def calibration(context, preprocessed_data, dask_client):
    ...
```

Now your step will be parallelized if it can be done by dask. 

If not needed, simply omit it.

### Adding your own step

To add a custom step:

```python
@registry.step(
    name="my_custom_step",
    inputs=(FromStep("simulation", "iwu_temporal_estimates"),),
    outputs=("custom_output",),
)
def my_custom_step(context, iwu_temporal, dask_client):
    # import your custom method and pass the same args as above.
    # your method should return something like this
    # return {"custom_output": iwu_temporal.mean()} 
    # or
    # return iwu_temporal.mean()
    # you can also get the config variable `some_parameter` as defined in the 
    # next section like this:
    # some_param = context.some_parameter
```

Add configuration (optional)

```yaml
my_custom_step:
  some_parameter: 42
```

Use the output downstream

```python
FromStep("my_custom_step", "custom_output")
```

That’s it — no pipeline wiring required.

### How everything fits together

In summary:

- Steps are registered via `@registry.step`
- Dependencies are declared via `FromStep`
- Execution order is inferred **automatically**
- Configuration is injected **dynamically**
- Outputs are stored and reused

This allows the pipeline to be:

- extensible
- declarative
- reproducible
- easy to reason about

## Run the pipeline

To run the pipeline:

```bash
python -m src/irrigation-processor/main.py
```

from the root of this project. 

Or

```python
from pathlib import Path
from irrigation_processor.main import execute_pipeline

execute_pipeline(
    config_file=Path("config.yml")
)
```

You can modify the xcube storage to `s3` if needed. 
Currently, `file` data storage from xcube is used as default.
The output directory for this default data store is `output_irrigation`

To use the s3 storage, do this in the `main.py`:

```python
storage = XcubeDataStoreStorage(
    "s3",
    store_kwargs=dict(
        storage_options=dict(
            anon=False,
            key=os.getenv("XCUBE_AWS_ACCESS_KEY_ID"),
            secret=os.getenv("XCUBE_AWS_SECRET_ACCESS_KEY"),
            client_kwargs=dict(
                endpoint_url=os.getenv("XCUBE_AWS_ENDPOINT_URL"),
            ),
        ),
        root=os.getenv("XCUBE_BUCKET_NAME")",
        max_depth=5,
    ),
)
```

Make sure you add the AWS creds to the `.env` file in the root folder.

```.dotenv
# The following are for you xcube data storage
XCUBE_AWS_ACCESS_KEY_ID=
XCUBE_AWS_SECRET_ACCESS_KEY=
XCUBE_AWS_ENDPOINT_URL=
XCUBE_BUCKET_NAME=
# The following are for CDSE S3 access to CLMS data as shown above
CDSE_AWS_ACCESS_KEY_ID=
CDSE_AWS_SECRET_ACCESS_KEY=
CDSE_AWS_ENDPOINT_URL=
```

The pipeline will:

- build the dependency graph
- execute required steps
- reuse stored results where available
- write final outputs to the configured data store

The pipeline produces:

- calibrated soil-moisture inversion parameters
- irrigation water use estimates (temporal & spatial)
- postprocessed irrigation datasets ready for analysis

All outputs are stored using a xcube data store.

**NOTE: This would be soon released as a python package.**

## Irrigation Processor Pipeline steps

### 1. Data Loader (`dataloader`)

Loads all required input datasets, such as:

- soil moisture observations (CLMS)
- land-cover maps (pre-computed available via deepESDL S3 bucket)
- meteorological data (e.g. ERA5 precipitation and evaporation) (CDS)

### 2. Preprocessing (`preprocessing`)

Prepares the raw datasets by:

- spatial and temporal alignment
- variable selection and transformation
- filtering invalid or unused variables
- combining these datasets into one

The result is a preprocessed dataset ready for modeling.

### 3. Calibration (`calibration`)

Estimates soil-moisture inversion parameters per grid cell.

Model parameters

The calibration step estimates four parameters:

| Parameter | Meaning (conceptual)                 |
| --------- | ------------------------------------ |
| `a`, `b`  | Nonlinear soil response coefficients |
| `z`       | Soil water storage scaling           |
| `RF`      | Rainfall–soil moisture coupling      |

Calibration is:

- performed independently for each pixel using `scipy.optimize`
- NaN-safe

The result is a spatial dataset of calibration parameters ready for simulation.

### 4. Simulation (`simulation`)

Uses the calibrated parameters to simulate irrigation water use over time.

Conceptually:

- soil moisture changes + meteorological forcing -> irrigation signal
- irrigation is aggregated to weekly and biweekly scales
- thresholds are applied to remove noise and unrealistic values

Outputs:

- temporal IWU estimates (for time-series based analysis)
- spatial IWU estimates (for grid-based visualization)

### 5. Postprocessing (`postprocessing`)

Applies final filters to the simulated irrigation estimates:

- temporal masking (e.g. specific months)
- spatial masking (e.g. irrigated areas only, we use this data: https://data.apps.fao.org/catalog/iso/f79213a0-88fd-11da-a88f-000d939bc5d8)

Produces the final irrigation water use datasets in both spatial and temporal 
chunks.


## How calibration works (conceptual)

Calibration solves an inverse problem:

    Given soil moisture dynamics and meteorological forcing, estimate 
    parameters that best explain observed changes.

Key properties:

- solved independently per pixel
- robust to missing data
- produces exactly four parameters per cell

Diagnostic statistics (e.g. number of unique parameter sets) can optionally be logged.

### How simulation works (conceptual)

Simulation uses the calibrated parameters to:

- estimate irrigation contributions at each time step
- remove small or spurious signals
- aggregate to meaningful temporal scales
- clip unrealistic values

This produces physically consistent irrigation estimates suitable for analysis.



### FAQ / Troubleshooting

#### The pipeline exits early without recomputing data

This is expected behavior.
If outputs already exist in the data store, steps are skipped automatically.

Delete the relevant dataset IDs to force recomputation.

#### Why are results zero in some regions?

Common causes:

- temporal masking excluded those months
- spatial mask threshold removed the area
- precipitation signal too weak

Inspect postprocessing configuration.