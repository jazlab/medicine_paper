# MEDICINE: Motion Estimation by DIstributional Contrastive Inference for NEurophysiology

This repository contains code to reproduce all main test  results in the
following paper: MEDICINE

## Reproducing MEDiCINe Paper Figures

This repository contains code to generate all the main text results in the
MEDiCINe paper. Each notebook in the [figures](./figures) directory produces one
of these results. All data needed to generate these is available for download at
[https://osf.io/acu8j/files/osfstorage#](https://osf.io/acu8j/files/osfstorage#).
You can reproduce the MEDiCINe paper results by taking the following steps:

First, clone this repository by navigating to your local target directory and
running:
```
$ git clone https://github.com/jazlab/medicine_paper.git
```

Second, install the dependencies with:
```
pip install -r requirements.txt
```
It is recommended to create and install the requirements in a virtual
environment.

Third, download all the data files from
[https://osf.io/acu8j/files/osfstorage#](https://osf.io/acu8j/files/osfstorage#),
move them to [cache](./cache), and unzip the zip files.

Lastly, run each of the Jupyter notebooks in [figures](./figures).

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
[simulated_data/generate_data](./simulated_data/generate_data). Due to the
prohibitively large total size of the simulated datasets (~23Tb), we have not
open-sourced these raw datasets.

### Running Motion Estimation on Simulated Datasets

If you would like to run motion estimation methods on our simulated datasets,
follow the instructions in
[simulated_data/motion_estimation](./simulated_data/motion_estimation).

### Spike Sorting Simulated Datasets

If you would like to run spike sorting on motion-corrected simulated datasets,
follow the instructions in
[simulated_data/spike_sorting](./simulated_data/spike_sorting).
