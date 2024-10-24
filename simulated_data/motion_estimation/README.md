# Running Motion Estimation on Simulated Datasets

This library is used to run motion estimation methods on simulated datasets and
compute errors of the results.

Running these methods is computationally expensive, so we recommend downloading
the motion estimation results from our [OSF data
repository](https://osf.io/acu8j/files/osfstorage) instead of generating the
datasets.

Available motion estimators are:
* `kilosort_datashift`. This runs the
  [datashift.py](https://github.com/MouseLand/Kilosort/blob/main/kilosort/datashift.py)
  function for motion estimation in Kilosort4.
* `dredge`. This runs the official DREDge motion estimation with default
  parameters using SpikeInterface.
* `rigid_fast`. This runs a version of the DREDge motion estimation with rigid
  motion (not varying in depth) method using SpikeInterface.
* `medicine`. This runs the official MEDiCINe motion estimation with default
  parameters.
* `medicine_rigid`. This runs a version of MEDiCINe with rigid motion (not
  varying in depth).

## Usage: Running Motion Estimators

To run all motion estimators on all datasets, run the following:
```
$ python run_motion_estimators.py
```

Note that running `dredge` and `rigid_fast` require the full recording to be
present in the dataset directory. Due to dataset size constraints, out
open-sourced simulated data repository contains only extracted spikes, not the
full recording, so if you want to run `dredge` or `rigid_fast`, you must
generate the raw recording. See `../generate_data` for code and documentation to
do this.

To run only a subset of the estimators on all datasets, you can use the
`estimator_names` keyword argument with a comma-separated list of estimator
names. For example, to run all the estimators except `dredge` and `rigid_fast`,
run:
```
$ python run_motion_estimators.py estimator_names=kilosort_datashift,medicine,medicine_rigid
```

To run only a subset of the datasets, you can use the dataset hyperparameters as
keyword arguments to filter the set of datasets. For example, to run only
MEDiCINe on only the datasets with non-rigid motion, run:
```
$ python run_motion_estimators.py estimator_names=medicine non_rigidity=0.5
```

## Usage: Computing Motion Estimation Errors

To compute the errors for all motion estimators on all datasets, run the
following:
```
$ python run_compute_errors.py
```

Note that you must have first run or downloaded the motion estimation results
for all methods on all datasets into `../cache/sim_motion_estimation`.

Just list `run_motion_estimators.py`, the `run_compute_errors.py` script can
take arguments for filtering the methods and datasets to evaluate. For example,
to compute errors only for MEDiCINe on only the datasets with non-rigid motion,
run:
```
$ python run_compute_errors.py estimator_names=medicine non_rigidity=0.5
```
