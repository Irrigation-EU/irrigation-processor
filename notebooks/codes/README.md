## Notebooks

The notebooks in this directory were used for **exploration, prototyping, and step-by-step development** of the irrigation processing pipeline.

They are organized as numbered notebooks (`01` through `05`) to reflect the logical progression of the pipeline:

1. **01** – Data loadeing from CLMS and CDS data sources
2. **02** – Preprocessing these datasets  
3. **03** – Running calibration algorithm on the preprocessed dataset
4. **04** – Running simulation on the calibration parameters and preprocessed dataset 
5. **05** – Postprocessing the simulated estimates spatially and temporally

The production-ready code from these notebooks has since been **refactored and consolidated into the `ops` submodule** of this package.  
Each module in `ops` corresponds to one of the numbered steps above and is what actually executes when the pipeline is run.

The notebooks are kept for reference and reproducibility but are **not part of the runtime pipeline**.
