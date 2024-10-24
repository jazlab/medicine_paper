# MEDiCINe: Motion Estimation by Distributional Contrastive Inference for Neurophysiology

This repository contains code to reproduce all main test results in the
following paper:

MEDiCINe: Motion Estimation by Distributional Contrastive Inference for
Neurophysiology. Watters, N., Bussino, A., & Jazayeri, M. arXiv, 2024.

For more information about MEDiCINe, including demos and instructions for using
it on your own datasets, please visit the [MEDiCINe
website](https://jazlab.github.io/medicine/).

## Reproducing MEDiCINe Paper Figures

This repository contains code to generate all the main text results in the
MEDiCINe paper. Each notebook in the [./figures/](./figures) directory produces
one of these results. All data needed to generate these is available for
download at [https://figshare.com/projects/MEDiCINe/225027](https://figshare.com/projects/MEDiCINe/225027).
You can reproduce the MEDiCINe paper results by taking the following steps:

1. Clone this repository by navigating to your local target directory and
running `$ git clone https://github.com/jazlab/medicine_paper.git`

1. Install the dependencies with `$ pip install -r requirements.txt`. It is
   recommended to use a virtual environment with python version 3.10 or later.

2. Visit [https://figshare.com/projects/MEDiCINe/225027](https://figshare.com/projects/MEDiCINe/225027).
You should see one dataset there with an 8Gb file `data.zip`. Download this file. This may take several minutes. Once downloaded, un-zip `data.zip`. The contents should
be directories `nhp_datasets`, `sim_motion_estimation`, `simulated_datasets`,
`spike_sorting`, and `spike_sorting_ground_truth`. Move these contents to the
[./cache/](./cache) directory in your cloned repo.

1. Run each of the Jupyter notebooks in [./figures/](./figures).

## Generating Intermediate Data

The instructions above allo you to download intermediate data formats (e.g.
motion-estimation results) and generate results from those. If you would like to
generate the intermediate data formats themselves, we provide software for doing
that for the simulated datasets. You can (i) generate the simulated datasets,
(ii) run motion estimation on the simulated datasets, and (iii) run spike
sorting on the motion-corrected simulated datasets.

### Generating Simulated Datasets

If you would like to generate the recordings in our simulated dataset suite,
follow the instructions in
[./simulated_data/generate_data/](./simulated_data/generate_data). Due to the
prohibitively large total size of the simulated datasets (~23Tb), we have not
open-sourced these raw datasets.

### Running Motion Estimation on Simulated Datasets

If you would like to run motion estimation methods on our simulated datasets,
follow the instructions in
[./simulated_data/motion_estimation/](./simulated_data/motion_estimation).

### Spike Sorting Simulated Datasets

If you would like to run spike sorting on motion-corrected simulated datasets,
follow the instructions in
[./simulated_data/spike_sorting/](./simulated_data/spike_sorting).

## Reference

If you use MEDiCINe or a derivative of it in your work, please cite it as
follows:

```
@article{watters2024,
author = {Nick Watters and Alessio Buccino and Mehrdad Jazayeri},
title = {MEDiCINe: Motion Estimation by DIstributional Contrastive Inference for NEurophysiology},
url = {https://arxiv.org/},
journal = {arXiv preprint arXiv:},
year = {2024}
}
```
