
# WoundHealingAnalysis

**An open-source image analysis toolkit for wound-healing experiments.**

`WoundHealingAnalysis` is a Julia package designed to analyze microscopy images from _in vitro_ wound-healing (scratch) assays. It aims to provide an accessible and reproducible alternative to proprietary image analysis software, allowing researchers to process experimental images, segment wound regions, track cell movement, and quantify wound closure. All images tested came from the Cytation 10 microscope.

## Authors and contributors

Main authors:

- Carla M. Soprano --- Instituto de Física de Líquidos y Sistemas Biológicos (IFLySiB), Argentina.


---

## Features

The package provides tools for:

- **Image segmentation:** preprocessing microscopy images and identifying wound regions.
- **Visualization:** generating images to inspect intermediate processing steps and final results.
- **Quantitative analysis:** measuring wound area, wound width, and roughness.

*The available features and their configuration options may evolve as the package develops.*

---

## Installation

### Requirements

- [Julia](https://julialang.org/downloads/) (version **1.13.3** or later).
- A computer running Linux or Windows.

### Setting up the project

1. Download or clone this repository.
2. Open a terminal and navigate to the project directory.
3. Install the required Julia packages:

```bash
julia
using Pkg
Pkg.activate(".")
Pkg.instantiate()
using WoundHealingAnalysis
```

---

## Getting started

You do not need to know Julia or modify the source code to run the standard analysis workflows. The project provides command-line scripts and configuration files to customize the analysis.

### 1. Organize your experimental data

Keep your experimental images in a separate directory from the software.

For example:

```text
Microscopy/
├── WoundHealingAnalysis/       # This repository (project directory)
│   ├── configs
│   │    ├── PreprocessingSteps.toml
│   │    ├── Hyperparams.toml
│   │    └── configs.toml
│   ├── ...
│
└── WoundHealingExp/            # Your experimental data
    ├── yourconfigs
    │        ├── YourPreprocessingSteps.toml
    │        ├── YourHyperparams.toml
    │        └── yourconfigs.toml
    ├── day1/
    │   ├── image_001.tif
    │   └── image_002.tif
    ├── day2/
    │   ├── image_001.tif
    │   └── image_002.tif
    └──
```

Your data can be stored anywhere on your computer. You do not need to copy your experimental images into the project directory.

### 2. Configure your experiment

Before running an analysis, edit the appropriate configuration file (`config.toml`).

The configuration file specifies where your images are stored, how they are organized, and which image sequences should be analyzed.

For image segmentation, you can also adjust the parameters in `Hyperparams.toml` and select the processing steps in `SegmentationPipeline.toml`.

**Tip:** Configuration files inside the `examples/` directory provide working examples that you can use as a starting point.

See our protocol [Configuration guide](#configuration-guide) for more details.

### 3. Run the analysis

Open a terminal in the project directory and execute:

```
run_segmentation_playground(image_path=”/Microscopy/WoundHealingExp/your_image.tif”, hyperparams_path="/Microscopy/WoundHealingExp/yourconfigs/YourHyperparameters.toml”, preprocessing_path=”/Microscopy/WoundHealingExp/yourconfigs/YourPreprocessingSteps.toml”)
```

(if running on Windows, write raw before each path, as in image_path=raw"/Microscopy/WoundHealingExp/your_image.tif").

The playground allows you to inspect the effects of different processing steps and parameters before running the analysis on your complete dataset.

To run the pipeline on all of your files according to yourconfig.toml, execute:

```
run_segmentation_pipeline("/Microscopy/WoundHealingExp/yourconfigs/yourconfig.toml”, "/Microscopy/WoundHealingExp/yourconfigs/YourHyperparameters.toml”, "/Microscopy/WoundHealingExp/yourconfigs/YourPreprocessingSteps.toml”)
```

(same as before, add raw before each path if running on Windows).

**Tip:** You can drag and drop your files on the terminal in order to paste their respective paths.

---

## Configuration guide

Follow the instructions to select the configuration file and start processing your images.

The analysis is controlled through TOML configuration files. This allows you to change experimental paths, image naming conventions, and segmentation parameters without modifying the Julia source code.

| Configuration file | Purpose |
|---|---|
| `config.toml` | Experimental data locations, image sequences, and analysis settings |
| `Hyperparams.toml` | Parameters controlling image segmentation |
| `SegmentationPipeline.toml` | Processing steps and their execution order |

For detailed explanations of each parameter and examples of different configurations, see the files in `examples/`.

---

## Output

The analysis generates processed images and data files in the output directories specified by the configuration.

Depending on the selected workflow, outputs may include:

- Segmented images and intermediate processing results.
- .csv files with: wound area (scratch_report.csv), width distribution (width_report.csv for mean values and roughness_report.csv for their respective standard deviations) and times for each frame (time_report.csv is calculated from the images' metadata saying when it was last modified, relative to the first timepoint).

---

## Examples

The `examples/` directory contains example configuration files and usage examples on our experiments (see our protocol) to help you get started.

These examples are intended to illustrate how to organize your data and configure the analysis for different experimental setups.


---

## Citation

If you use `WoundHealingAnalysis` in your research, please cite:

**[Add citation once a paper or software reference is available.]**

---

## License

WoundHealingAnalysis.jl is Copyright (C) 2026 by the authors. See `LICENSE` for details.

This is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
