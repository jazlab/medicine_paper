"""Run sorting.

This script runs spike sorting using Kilosort4 on all datasets with all motion
estimation methods, and saves the results to ../../cache/spike_sorting.
"""

import shutil
from pathlib import Path

import get_datasets
import numpy as np
from spikeinterface import sorters as si_sorters
from spikeinterface.extractors import read_mearec
from spikeinterface.sortingcomponents.motion import motion_utils
from spikeinterface.sortingcomponents.motion.motion_interpolation import (
    InterpolateMotionRecording,
)

SIMULATED_DATASETS_DIR = Path("../../cache/simulated_datasets")
SPIKE_SORTING_DIR = Path("../../cache/spike_sorting")
MOTION_ESTIMATION_DIR = Path("../../cache/sim_motion_estimation")
ESTIMATOR_NAMES = [
    "kilosort_datashift",
    "dredge",
    "rigid_fast",
    "medicine",
    "medicine_rigid",
]


def run_spike_sorting(
    dataset_name: str,
    estimator_name: str,
    ks_type: str = "kilosort4",
):
    """Run spike sorting on dataset_name with motion estimation.

    Args:
        dataset_name: Name of dataset.
        estimator_name: Name of motion estimator.
        ks_type: Type of Kilosort to run.
    """
    print(
        f"\n\ndataset_name = {dataset_name}, estimator_name = {estimator_name}"
    )

    # Load recording
    recording_dir = SIMULATED_DATASETS_DIR / dataset_name
    mearec_filename = recording_dir / "recording.h5"
    if not mearec_filename.exists():
        raise FileNotFoundError(
            f"No recording found at {mearec_filename}. Be sure to generate the "
            "simulated datasets before extracting ground truth spike sorting. "
            "See ../generate_data for more information."
        )
    print(f"Loading recording {mearec_filename}")
    recording, _ = read_mearec(mearec_filename)

    # Load estimated motion
    motion_estimation_dir = (
        MOTION_ESTIMATION_DIR / dataset_name / estimator_name
    )
    motion = np.load(motion_estimation_dir / "motion.npy")
    if len(motion.shape) == 3:
        if motion.shape[0] == 1:
            motion = motion[0]
        elif motion.shape[1] == 1:
            motion = motion[:, 0]
        else:
            raise ValueError(f"Unknown motion shape {motion.shape}")
    time_bins = np.load(motion_estimation_dir / "time_bins.npy")
    depth_bins = np.load(motion_estimation_dir / "depth_bins.npy")
    if len(time_bins.shape) == 2:
        time_bins = time_bins[0]

    # Expand motion by one timestep to avoid boundary effect errors
    motion = np.concatenate([motion, motion[-1:]], axis=0)
    time_bins = np.concatenate(
        [time_bins, [time_bins[-1] + time_bins[1] - time_bins[0]]]
    )

    # Interpolate motion
    motion_object = motion_utils.Motion(
        displacement=motion,
        temporal_bins_s=time_bins,
        spatial_bins_um=depth_bins,
    )
    recording_corrected = InterpolateMotionRecording(
        recording,
        motion_object,
        border_mode="force_extrapolate",
    )

    # Create write directory
    write_dir = SPIKE_SORTING_DIR / dataset_name / estimator_name
    write_dir.mkdir(parents=True, exist_ok=True)

    # Get sorting parameters
    print("\nGetting parameters")
    params = si_sorters.get_default_sorter_params(ks_type)
    params["do_correction"] = False

    # Run spike sorting
    print("\nRunning kilosort")
    kilosort_output_folder = write_dir / f"kilosort_outputs"
    if (kilosort_output_folder / "spike_times.npy").exists():
        print(
            f"Kilosort output folder {kilosort_output_folder} already exists. "
            "Skipping."
        )
        return
    if kilosort_output_folder.exists():
        shutil.rmtree(kilosort_output_folder)
    sorting = si_sorters.run_sorter(
        sorter_name=ks_type,
        recording=recording_corrected,
        folder=kilosort_output_folder,
        docker_image=False,
        verbose=True,
        **params,
    )
    print(f"\nSaving sorting to {write_dir}")
    sorting_save_folder = write_dir / f"sorting_save"
    sorting = sorting.save(folder=sorting_save_folder, overwrite=True)


def main():
    """Run spike sorting on all sorting datasets."""
    dataset_names = get_datasets.get_dataset_names()
    for dataset_name in dataset_names:
        for estimator_name in ESTIMATOR_NAMES:
            run_spike_sorting(dataset_name, estimator_name)


if __name__ == "__main__":
    main()
