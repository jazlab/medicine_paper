"""Run ground truth spike train extractor.

This script extracts spike trains and template depths from simulated datasets.
It is used to generate ground truth for spike sorting.
"""

import shutil
from pathlib import Path

import get_datasets
import MEArec
import numpy as np

SIMULATED_DATASETS_DIR = Path("../../cache/simulated_datasets")
SPIKE_SORTING_GROUND_TRUTH_DIR = Path("../../cache/spike_sorting_ground_truth")


def extract_spike_trains(recording_dir: Path) -> None:
    """Extract spike trains and template depths from recording directory.

    Args:
        recording_dir: Path to recording directory.
    """
    # Load recording
    recording_path = recording_dir / "recording.h5"
    if not recording_path.exists():
        raise FileNotFoundError(
            f"No recording found at {recording_path}. Be sure to generate the "
            "simulated datasets before extracting ground truth spike sorting. "
            "See ../generate_data for more information."
        )
    recording = MEArec.load_recordings(
        recording_path, load=["spiketrains", "templates"], load_waveforms=False
    )

    # Extract spike trains and template locations
    spike_trains = [np.array(x) for x in recording.spiketrains]
    template_depths = np.array(recording.template_locations)[:, :, 2]

    # Save results
    write_dir = SPIKE_SORTING_GROUND_TRUTH_DIR / recording_dir.name
    if write_dir.exists():
        shutil.rmtree(write_dir)
    spike_trains_dir = write_dir / "spike_trains"
    spike_trains_dir.mkdir(parents=True)
    for i, spike_train in enumerate(spike_trains):
        np.save(spike_trains_dir / f"{i}.npy", spike_train)
    np.save(write_dir / "template_depths.npy", template_depths)
    print(f"\nSaved spike trains and template depths to {write_dir}")


def main():
    """Run ground truth spike train extractor on all sorting datasets."""
    dataset_names = get_datasets.get_dataset_names()
    for dataset_name in dataset_names:
        print(f"\nRunning recording {dataset_name}")
        recording_dir = SIMULATED_DATASETS_DIR / dataset_name
        extract_spike_trains(recording_dir=recording_dir)

    print("\n\nDONE\n\n")


if __name__ == "__main__":
    main()
