# Running Spike Sorting on Simulated Datasets

This library is used to run spike sorting on simulated datasets using the
different motion estimation methods.

Running spike sorting is computationally expensive, so we recommend downloading
the spike sorting results from our [OSF data
repository](https://osf.io/acu8j/files/osfstorage) instead of running thise
code. However, if you would like to run spike sorting yourself, follow the
instructions below.

## Usage: Running Kilosort4 Spike Sorting

First note that we only ran spike sorting on a random subset of 40 of our 384
simulated datasets, to avoid excessive use of computational resources. To view
this set of 40, run
```
$ python get_datasets.py
```
This will print a list of 40 datasets.

To run Kilosort4 spike sorting all motion estimators on these datasets, run the
following:
```
$ python run_sorting_ks4.py
```
This uses SpikeInterface to run Kilosort3 will produce spike sorting results in
the `../../cache/spike_sorting` directory. This takes many hours to run to
completion.

Note that this requires the raw recording files. See
`../generate_data/README.md` for instructions on how to generate those.

## Usage: Computing Ground Truth Spike Sorting

To compute the ground truth spike sorting, run
```
$ python run_ground_truth.py
```
This extracts the ground truth spike trains from the 40 datasets used for
sorting and saves the results to `../../cache/spike_sorting_ground_truth`.

Note that this also requires the raw recording files. See
`../generate_data/README.md` for instructions on how to generate those.
