# MEDiCINe: Motion Estimation by Distributional Contrastive Inference for Neurophysiology

This repository contains code to reproduce all main results in the following
paper:

MEDiCINe: Motion Correction for Neural Electrophysiology Recordings. Watters,
N., Bussino, A., & Jazayeri, M. arXiv, 2024.

For more information about MEDiCINe, including demos and instructions for using
it on your own datasets, please visit the [MEDiCINe
website](https://jazlab.github.io/medicine/).

## Reproducing MEDiCINe Paper Figures

This repository contains code to generate all the main text results in the
MEDiCINe paper. Each notebook in the [./figures/](./figures) directory produces
one of these results. All data needed to generate these is available for
download at
[https://osf.io/acu8j/files/osfstorage](https://osf.io/acu8j/files/osfstorage).
You can reproduce the MEDiCINe paper results by taking the following steps:

1. Clone this repository by navigating to your local target directory and
running `$ git clone https://github.com/jazlab/medicine_paper.git`

1. Install the dependencies with `$ pip install -r requirements.txt`. It is
   recommended to use a virtual environment with python version 3.10 or later.
   At this point, to ensure the MEDiCINe package is installed and working, you
   can run `$ python -m medicine_demos.run_demo`, which should run without error
   and produce several plots.

2. Visit
[https://osf.io/acu8j/files/osfstorage](https://osf.io/acu8j/files/osfstorage).
You should see five zip files. Download all of these, which will total ~9Gb.
This may take several minutes, and we recommend using a high-speed ethernet
connection. Once downloaded, un-zip them all and move them to the
[./cache/](./cache) directory in your cloned repo. At this point, the
[./cache/](./cache) directory in your cloned repo should contain directories
`nhp_datasets`, `sim_motion_estimation`, `simulated_datasets`, `spike_sorting`,
and `spike_sorting_ground_truth`.

1. Run each of the Jupyter notebooks in [./figures/](./figures).

## Generating Intermediate Data

The instructions above allow you to download intermediate data formats (e.g.
motion estimation results) and generate results from those. If you would like to
generate the intermediate data formats themselves, we provide software for doing
that for the simulated datasets. You can generate the simulated datasets, run
motion estimation on the simulated datasets, and run spike sorting on the
motion-corrected simulated datasets. Each of these is computationally expensive,
so if you are only interested in reproducing the results then we recommend
following the instructions above.

### Generating simulated datasets

If you would like to generate the recordings in our simulated dataset suite,
follow the instructions in
[./simulated_data/generate_data/](./simulated_data/generate_data). Due to the
prohibitively large total size of the simulated datasets (~23Tb), we have not
open-sourced these raw datasets.

### Running motion estimation on simulated datasets

If you would like to run motion estimation methods on our simulated datasets,
follow the instructions in
[./simulated_data/motion_estimation/](./simulated_data/motion_estimation).

### Spike sorting simulated datasets

If you would like to run spike sorting on motion-corrected simulated datasets,
follow the instructions in
[./simulated_data/spike_sorting/](./simulated_data/spike_sorting).

## Reference

If you use MEDiCINe or a derivative of it in your work, please cite it as
follows:

```
@article {Watters2024.11.06.622160,
	author = {Watters, Nicholas and Buccino, Alessio P and Jazayeri, Mehrdad},
	title = {MEDiCINe: Motion Correction for Neural Electrophysiology Recordings},
	elocation-id = {2024.11.06.622160},
	year = {2024},
	doi = {10.1101/2024.11.06.622160},
	URL = {https://www.biorxiv.org/content/10.1101/2024.11.06.622160},
	eprint = {https://www.biorxiv.org/content/10.1101/2024.11.06.622160.full.pdf},
	journal = {bioRxiv}
}
```
