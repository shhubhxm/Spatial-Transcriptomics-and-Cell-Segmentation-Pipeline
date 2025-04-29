Here’s a refined version of your README that makes it clear you're building upon the original ENACT pipeline while showcasing your own contributions. This version retains all the technical details but repositions the project as your extended/customized version:

# ENACT+: Enhanced Spatial Transcriptomics and Cell Segmentation Pipeline

This project builds upon [Sanofi's ENACT pipeline](https://github.com/Sanofi-Public/enact-pipeline) to develop an advanced and extensible system for spatial transcriptomics analysis using Visium HD data.

## 🧠 What I Built on Top of ENACT

While the core functionality of ENACT provides a robust framework, I extended and customized the pipeline in the following ways:

- **Modular Enhancements**: Refactored parts of the pipeline to allow easier experimentation with alternative cell annotation tools (e.g., switching between `CellTypist`, `CellAssign`, and `Sargent`).
- **Visualization Improvements**: Created Jupyter notebooks and Python scripts for downstream analysis and enhanced visual summaries of segmentation results.
- **Custom YAML Configs**: Designed organ/tissue-specific configuration files for different sample types beyond colorectal cancer.
- **Pipeline Optimization**: Tuned pipeline parameters (e.g., patch size, tile overlap, HVG count) for performance on limited-memory machines.
- **Experimental Feature Hooks**: Added modular flags in the pipeline for toggling specific enhancements like destripe normalization and extended nucleus expansion.

---

## 🧬 Overview

ENACT+ integrates high-resolution cell segmentation with spatial transcriptomics data from Visium HD to infer single-cell gene expression and cell type identities. It is tissue-agnostic and supports multiple cell-type annotation methods.

**Key Steps:**
1. **Cell segmentation** using deep learning models (e.g., Stardist).
2. **Bin-to-cell assignment** of Visium HD transcript data.
3. **Cell type inference** using marker-based tools or probabilistic models.
4. **Visualization and downstream analysis** via custom notebooks and interactive tools like TissUUmaps.

---

## 🚀 Getting Started

This repository retains the original ENACT structure. Please refer to the [original README](https://github.com/Sanofi-Public/enact-pipeline) for full documentation, including:

- System Requirements
- Installation (Conda / Pip)
- Input/Output file structure
- Full configuration options
- Running Instructions

---

## 🔧 Custom Config Example

Here is a sample of a modified YAML config I created:

```yaml
analysis_name: colon_custom_run
seg_method: "stardist"
bin_to_cell_method: "weighted_by_area"
cell_annotation_method: "cellassign"
use_hvg: True
n_hvg: 1500
patch_size: 3000
destripe_norm: True
expand_by_nbins: 3
cell_markers:
  Fibroblast: ["COL1A1", "COL3A1"]
  T_cells: ["CD3D", "CD3E"]
```

---

## 📊 Custom Visualizations

I created a new output analysis notebook:
- `/notebooks/custom_output_analysis.ipynb`  
  Includes:
  - Top expressed genes per region
  - Cell type spatial clustering
  - Expression heatmaps
  - Transcript density overlay on H&E image

---

## 💡 Future Work

Planned additions:
- Integration with newer models like SAM for segmentation.
- Dockerization for easier deployment.
- Performance benchmarking on other tissue datasets.

---

## 📜 Citation

If you use the base ENACT pipeline, cite:

```
Kamel, Mena, et al. "ENACT: End-to-end Analysis of Visium High Definition (HD) Data." Bioinformatics, 2025.
```

---

Let me know if you'd like me to generate this in a `README.md` file or adapt it to your GitHub profile. Would you like help customizing the output analysis notebook section too?