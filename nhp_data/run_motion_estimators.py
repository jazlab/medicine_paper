"""Run motion estimators."""

import shutil
import sys
from pathlib import Path

import numpy as np
import spikeinterface.extractors as se
import spikeinterface.preprocessing as spre

sys.path.append("../../motion_estimation")
import run_kilosort
import run_medicine
import run_spikeinterface

sys.path.append("../peak_extraction")
import get_neuropixels_probe

_DATASETS_DIR = Path("../../cache/monkey_data/recordings")
_PEAKS_DIR = Path("../../cache/monkey_data/peaks")

_ESTIMATOR_NAMES = [
    # 'kilosort_like',
    # 'rigid_fast',
    # 'nonrigid_accurate',
    # 'nonrigid_accurate_large_steps',
    # 'medicine',
    # 'medicine_nonrigid',
    # 'kilosort_like',
    "kilosort_datashift",
    "rigid_fast",
    # 'nonrigid_accurate',
    # 'nonrigid_accurate_large_steps',
    "medicine",
    "medicine_nonrigid",
    "dredge",
]


def _run_motion_estimation(recording, peaks_dir):
    """Run motion estimators on recording."""

    # Create write directory for motion estimation
    write_dir = peaks_dir / "motion_estimation"

    # Run estimators
    for estimator_name in _ESTIMATOR_NAMES:
        print(f"\nRunning {estimator_name}")

        # Create estimator write_dir
        write_dir_estimator = write_dir / estimator_name
        write_dir_estimator.mkdir(parents=True, exist_ok=True)

        # Continue if already ran
        if (write_dir_estimator / "motion.npy").exists():
            # print(f'Already ran {estimator_name} on {peaks_dir}')
            # continue
            shutil.rmtree(write_dir_estimator)
            write_dir_estimator.mkdir(parents=True, exist_ok=True)

        # Run estimator on recording
        if estimator_name == "medicine":
            run_medicine.run_medicine(
                recording_dir=peaks_dir,
                write_dir=write_dir_estimator,
                training_steps=10000,
                motion_num_depth_bins=1,
                sampling_frequency=30000,
            )
        elif estimator_name == "medicine_nonrigid":
            run_medicine.run_medicine(
                recording_dir=peaks_dir,
                write_dir=write_dir_estimator,
                training_steps=10000,
                motion_num_depth_bins=2,
            )
        elif estimator_name == "kilosort_datashift":
            run_kilosort.run_kilosort(
                recording_dir=peaks_dir,
                write_dir=write_dir_estimator,
            )
        else:
            run_spikeinterface.run_spikeinterface(
                recording=recording,
                recording_dir=peaks_dir,
                write_dir=write_dir_estimator,
                estimator_name=estimator_name,
            )


def main(subject, session):
    # Read recording
    recording_filename = f"sub-{subject}_ses-{session}_ecephys.nwb"
    print(f"recording_filename = {recording_filename}")
    recording_path = _DATASETS_DIR / recording_filename
    print(f"recording_path = {recording_path}")
    peaks_dir = _PEAKS_DIR / f"{subject}_{session}"

    print("Reading recording")
    recording = se.read_nwb(
        recording_path,
        load_recording=True,
        load_sorting=False,
        electrical_series_path="acquisition/ElectricalSeriesAP",
    )

    # Hacky stuff to avoid the bug in spikeinterface that messes up the probe
    # and channels when reading from nwb files. Specifically, the recording does
    # not take into account the electrical seris acquisition specified.
    new_channel_ids = np.array(
        [str(i) for i in range(len(recording.channel_ids))], dtype="<U5"
    )
    np_probe = get_neuropixels_probe.get_neuropixels_probe()
    recording = recording.set_probe(np_probe)
    recording._main_ids = new_channel_ids

    # Bandpass filter and common reference
    print("Bandpass filtering")
    recording_f = spre.bandpass_filter(recording)
    print("Common referencing")
    recording_cmr = spre.common_reference(recording_f)

    # Run motion estimation
    print("\nRunning motion estimation")
    _run_motion_estimation(recording=recording_cmr, peaks_dir=peaks_dir)

    print("\n\nDONE\n\n")


if __name__ == "__main__":
    main(**dict(arg.split("=") for arg in sys.argv[1:]))
