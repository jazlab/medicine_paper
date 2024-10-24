"""Run spikeinterface motion estimation."""

import numpy as np
from spikeinterface.extractors import read_mearec
from spikeinterface.preprocessing import motion as sp_motion_lib
from spikeinterface.sortingcomponents.motion import estimate_motion

_SPIKEINTERFACE_ESTIMATORS = [
    "dredge",
    "rigid_fast",
]


def run_spikeinterface(
    recording, recording_dir, write_dir, estimator_name, random_seed=0
):
    if estimator_name not in _SPIKEINTERFACE_ESTIMATORS:
        raise ValueError(
            f"estimator_name is {estimator_name} but must be in "
            f"{_SPIKEINTERFACE_ESTIMATORS}"
        )

    # Set random seed
    np.random.seed(random_seed)

    # Load peaks and peak locations
    print(f"Loading peaks and peak locations from {recording_dir}")
    peaks = np.load(recording_dir / "peaks.npy")
    peak_locations = np.load(recording_dir / "peak_locations.npy")

    # Run motion estimation
    print(f"Estimating motion using {estimator_name}")
    estimate_motion_kwargs = sp_motion_lib.motion_options_preset[
        estimator_name
    ]["estimate_motion_kwargs"]
    motion_object = estimate_motion(
        recording, peaks, peak_locations, **estimate_motion_kwargs
    )
    motion = motion_object.displacement
    time_bins = motion_object.temporal_bins_s
    depth_bins = motion_object.spatial_bins_um

    # Save outputs
    print(f"Saving outputs to {write_dir}")
    np.save(write_dir / "time_bins.npy", time_bins)
    np.save(write_dir / "depth_bins.npy", depth_bins)
    np.save(write_dir / "motion.npy", motion)

    print("\n\nDONE\n\n")


def run_spikeinterface_from_mearec(
    recording_dir, write_dir, estimator_name, random_seed=0
):
    print(f"\n\nRunning {estimator_name} on recording {recording_dir.name}\n")

    # Load recording
    mearec_filename = recording_dir / "recording.h5"
    print(f"Loading recording {mearec_filename}")
    recording, _ = read_mearec(mearec_filename)

    # Run motion estimation
    run_spikeinterface(
        recording,
        recording_dir,
        write_dir,
        estimator_name,
        random_seed=random_seed,
    )
