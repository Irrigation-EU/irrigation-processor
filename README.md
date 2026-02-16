# 🌱 Irrigation Processor

[![CI](https://github.com/Irrigation-EU/irrigation-processor/actions/workflows/ci.yml/badge.svg)](https://github.com/Irrigation-EU/irrigation-processor/actions/workflows/ci.yml)
[![codecov](https://codecov.io/github/Irrigation-EU/irrigation-processor/graph/badge.svg?token=3KFCF7C6DA)](https://codecov.io/github/Irrigation-EU/irrigation-processor)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v0.json)](https://github.com/charliermarsh/ruff)
[![isort](https://img.shields.io/badge/imports-isort-1674b1.svg)](https://pycqa.github.io/isort/)
[![Pixi](https://img.shields.io/badge/env-pixi-5A67D8.svg)](https://pixi.sh)


Irrigation Processor is a scientific Python package for estimating irrigation 
water use (IWU) across **Europe** from soil moisture, meteorological variables, and land-cover 
data using a reproducible, multi-step processing pipeline.

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
  time_range: ["2020-01-01", "2020-12-31"]
  bbox: [-5.0, 43.0, -4.0, 44.0]
  use_gleam: false

# Individual step configurations
dataloader:
  cds_spatial_res: 0.1
  cds_variable_names: ["potential_evaporation", "total_precipitation"]

calibration:
  allowed_months: [4, 5, 6, 7, 8, 9]
  rainfall_threshold: 0.1
  check_calibration: true

simulation:
  # Optional: custom spatial chunking
  spatial_chunks:
    time: 1
    lat: 50
    lon: 50

postprocessing:
  temporal_allowed_months: [4, 5, 6, 7, 8, 9]
  spatial_mask_threshold: 0.5

# Dask parallelization configuration
dask:
  dask_kwargs:
    n_workers: 4
    threads_per_worker: 1
    memory_limit: "4GB"

# Storage configuration
storage:
  store_id: "s3"
  store_kwargs: ...
```

(keep in mind, the configuration file above is an example)

The `base` section contains shared configuration that is automatically merged 
into every step.

### Storage


You can modify the xcube storage to `s3` if needed. 
Currently, `file` data storage from xcube is used as default.
The output directory for this default data store is `output_irrigation`

To use the s3 storage, adjust the `storage` values in the `config.yml`:

```yaml
storage:
  store_id: "s3"
  store_kwargs:
    root: "<your-bucket-name>"
    max_depth: 5
    storage_options:
        anon: false
        key: "your-access-key"
        secret: "your-secret-key"
        client_kwargs:
          endpoint_url: "your-endpoint-url"
```

Make sure you also add the AWS creds to the `.env` file in the root folder.

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

### Advanced Configuration: Chunking and Dask

The pipeline supports advanced configuration for performance tuning, specifically for Dask parallelization and Xarray chunking.

#### Parallelization (`dask`)

The `dask` section allows you to configure the local Dask cluster used for processing:

- `n_workers`: Number of worker processes.
- `threads_per_worker`: Number of threads per worker.
- `memory_limit`: Memory limit per worker (e.g., "4GB").

#### Chunking

Individual steps support custom chunking for Xarray datasets to optimize memory usage and processing speed. This is typically configured via `*_chunks` keys:

- `dataloader`: `cds_intermediate_chunks`, `cds_final_chunks`.
- `preprocessing`: `swi_chunks`, `merged_chunks`.
- `simulation`: `spatial_chunks`.
- `calibration`: `calibration_chunks`.


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
def preprocessing(context, storage, sm_data_id, lc_data):
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

- is a Pydantic model (`AppConfig`)
- contains the merged config (base + step config)

`storage`

The second argument is the `storage` object, which provides the shared data store. You can use it to check what data already exists and load or save datasets.

### Parallelization

The pipeline automatically initializes a local Dask cluster with parameters provided in the `config.yml` under the `dask` key. Pipeline steps using Xarray with Dask will automatically leverage this cluster.

### Adding your own step

To add a custom step:

```python
@registry.step(
    name="my_custom_step",
    inputs=(FromStep("simulation", "iwu_temporal_estimates"),),
    outputs=("custom_output",),
)
def my_custom_step(context, storage, iwu_temporal):
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

The easiest way to run the pipeline is using [pixi](https://pixi.sh):

```bash
# Run with default config
pixi run irr-proc

# Run with custom config
pixi run irr-proc --config my_config.yml

# Visualize the pipeline DAG
pixi run irr-proc --visualize
```

Alternatively, run as a Python module:

```bash
python -m irrigation_processor.main --config config.yml
```

> Note:
> If you plan to use variables from the Gleam dataset, this pipeline currently 
> expects the dataset to be available in your selected xcube data store. 
> 
> Please download the dataset from the [here](https://www.gleam.eu/#downloads) and make sure it is 
> accessible in your configured data store before running the pipeline.
> 
> For e.g. if using file store, add it in the root folder of the xcube data 
> store, similarly for S3 store, add it to the bucket with the following name
> `gleamv4_2b.zarr`


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

- soil moisture observations (`xcube-clms data store`)
- land-cover maps (`xcube-cds data store`)
- meteorological data from ERA5-Land (`xcube-cds data store`) / Gleam (needs 
to be [downloaded](https://www.gleam.eu/#downloads), 
we only use `potential_evaporation` from gleam dataset)

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

- soil moisture changes + meteorological variables -> irrigation signal
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


    The parameters of the SM-Inversion are calibrated by optimizing the 
    model performances in properly reproducing occurred rainfall amounts. 
    To do this, the calibration is carried out during non-irrigation days 
    (e.g., winter and days with rainfall occurrence during the irrigation 
    season). 

Key properties:

- solved independently per pixel
- robust to missing data
- produces exactly four parameters per cell

Diagnostic statistics (e.g. number of unique parameter sets) can optionally be logged.

## How simulation works (conceptual)

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
- irrigation did not occur

Inspect postprocessing configuration.