# ENACT: Spatial Transcriptomics and Cell Segmentation Pipeline

Spatial transcriptomics (ST) enables researchers to study gene expression while preserving spatial context within tissue samples. A significant challenge has been the resolution limitations of sequencing-based ST platforms. With the advent of Visium High Definition (HD) technology, we can now analyze transcript data at near single-cell resolution.

**ENACT** is an end-to-end spatial transcriptomics analysis pipeline that integrates advanced cell segmentation with Visium HD data to infer transcript-level cell types across whole tissue sections. The pipeline is designed to be tissue-agnostic, modular, and highly scalable — making it suitable for use across a range of biomedical research applications.

![plot](figs/pipelineflow.png)

---

## What ENACT Does

- **Segments high-resolution tissue images** using neural network-based models (e.g., Stardist)
- **Aggregates VisiumHD bin-level transcripts** to form cell-specific expression profiles
- **Assigns cell types** using probabilistic and ML-based methods like `CellTypist`, `CellAssign`, or custom marker databases
- **Visualizes results interactively** with integrated support for TissUUmaps

This version of ENACT introduces enhancements that improve usability, performance, and downstream analytical flexibility, tailored to specific tissue types and experimental designs.

---

## Key Enhancements in This Version

- Reconfigured YAML-based interface for flexible tissue-specific runs  
- Streamlined support for alternate segmentation methods and cell typing tools  
- Optimized memory usage for chunk-wise processing of large WSIs  
- Developed rich visual analysis notebooks for downstream interpretation  
- Enabled destriping and normalization flags for finer control in preprocessing  

---

## Index

1. [Installation](#installation)
2. [Input & Output Structure](#input--output-structure)
3. [Configuration](#configuration)
4. [Running the Pipeline](#running-the-pipeline)
5. [Working With Results](#working-with-results)
6. [Visualization on TissUUmaps](#visualization-on-tissuumaps)
7. [Reproducing Results](#reproducing-results)
8. [Synthetic Data Support](#synthetic-data-support)
9. [Citation](#citation)

---

## Installation

ENACT requires Python 3.10 and optionally a CUDA-compatible GPU. Recommended system specs:

- **32-core CPU**, **64GB RAM**, **100GB disk** (for full-resolution slides)
- **Python**: 3.10  
- **Optional**: GPU w/ CUDA 11+

### Option 1: From Source

```bash
git clone https://github.com/<your-username>/enact-pipeline.git
cd enact-pipeline
make setup_py_env
```

Update the `Makefile` to point to your conda env location.

### Option 2: From PyPI

```bash
pip install enact-SO
```

---

## Input & Output Structure

### Required Inputs

- `tissue_image.btf`: High-resolution whole slide image
- `tissue_positions.parquet`: Bin positions from SpaceRanger
- `filtered_feature_bc_matrix.h5`: 2um-resolution gene-bin matrix from VisiumHD

### Output Directory

ENACT automatically stores all outputs under a `cache/` folder, organized by analysis name. Outputs include:

```
cache/
└── <analysis_name>/
    ├── chunks/
    │   ├── bins_gdf/
    │   ├── cells_gdf/
    │   └── results/
    ├── tmap/
    └── cells_df.csv
```

---

## Configuration

You can configure ENACT in two ways:

### Option 1: Python Class Interface

```python
from enact.pipeline import ENACT

run = ENACT(
    cache_dir="cache/",
    wsi_path="sample_image.btf",
    visiumhd_h5_path="filtered_feature_bc_matrix.h5",
    tissue_positions_path="tissue_positions.parquet",
    cell_annotation_method="celltypist"
)
```

### Option 2: YAML File

```yaml
analysis_name: colon_sample_run
cache_dir: ./cache
paths:
  wsi_path: path/to/image.btf
  visiumhd_h5_path: path/to/h5
  tissue_positions_path: path/to/positions.parquet
params:
  bin_to_cell_method: weighted_by_cluster
  cell_annotation_method: cellassign
  use_hvg: true
  n_hvg: 1000
```

---

## Running the Pipeline

You can run the pipeline using `make` or Python scripts:

```bash
make run_enact
```

Or run steps manually from your notebook/script using the ENACT class.

---

## 📊 Working With Results

A Jupyter notebook is provided to analyze the output:

- Load `cells_adata.csv` for cell coordinates and annotations
- Visualize top expressed genes
- Compute spatial clustering metrics
- Generate region-specific plots

---

## 🧭 Visualization on TissUUmaps

ENACT generates a `.tmap` project file for visualizing spatial results. To view:

1. Install [TissUUmaps](https://tissuumaps.github.io/)
2. Load the file from `cache/tmap/`
3. Explore spatial cell typing and transcript patterns interactively

![plot](figs/tissuumaps.png)

---

## Reproducing Results

To reproduce a benchmark run on 10X Genomics' Human Colorectal dataset:

```bash
make reproduce_results
```

This will download all necessary files and execute predefined config combinations (e.g., `weighted_by_area` + `cellassign`).

---

## Synthetic Data Support

You can also run ENACT on synthetic datasets (e.g., from Xenium or seqFISH+):

```yaml
run_synthetic: true
```

Then run:

```bash
make run_enact
```

Synthetic data notebooks are available under `/src/synthetic_data`.

---

## Citation

If you use this project or build on it for your own research, please cite:

```
@article{10.1093/bioinformatics/btaf094,
  title={ENACT: End-to-end Analysis of Visium High Definition (HD) Data},
  journal={Bioinformatics},
  year={2025},
  doi={10.1093/bioinformatics/btaf094}
}
```
