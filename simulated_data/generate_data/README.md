# Generating datasets

This library is used to generate simulated datasets with a variety of motion and
stability statistics, and to extract spikes from these datasets. Generating
these datasets is computationally expensive (both in time and memory), so we
recommend downloading the spike datasets from our [OSF data
repository](https://osf.io/acu8j/files/osfstorage) instead of generating the
datasets. Note that all motion estimation methods can be run on extracted spikes
alone without the raw data, so if you just want to reproduce our results, please
do not generate the datasets. The only reasons to generate the datasets would be
if you either (i) need the raw voltage data instead of extracted spikes, or (ii)
want to generate datasets with different parameters.

If you do want to generate the datasets, see the documentation in
`run_generate_data.py`.
