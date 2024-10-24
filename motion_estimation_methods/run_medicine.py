"""Run medicine."""

import numpy as np
import torch
from medicine import run as medicine_run


def run_medicine(
    recording_dir,
    write_dir,
    sampling_frequency=32000,
    motion_num_depth_bins=2,
    random_seed=0,
    **medicine_kwargs
):
    # Set random seed
    torch.random.manual_seed(random_seed)
    np.random.seed(random_seed)

    # Load peak depths data
    peak_locations = np.load(recording_dir / "peak_locations.npy")
    peak_locations = np.array([x.tolist() for x in peak_locations])
    peak_locations = dict(
        x=peak_locations[:, 0],
        y=peak_locations[:, 1],
        z=peak_locations[:, 2],
        alpha=peak_locations[:, 3],
    )
    peak_depths = peak_locations["y"]

    # Load peaks data
    peaks = np.load(recording_dir / "peaks.npy")
    peaks = np.array([x.tolist() for x in peaks])
    peaks = dict(
        sample_index=peaks[:, 0],
        channel_index=peaks[:, 1],
        amplitude=peaks[:, 2],
        segment_index=peaks[:, 3],
    )
    peak_amplitudes = peaks["amplitude"]
    peak_times = peaks["sample_index"] / sampling_frequency

    # Fit medicine model
    trainer = medicine_run.run_medicine(
        peak_times=peak_times,
        peak_depths=peak_depths,
        peak_amplitudes=peak_amplitudes,
        output_dir=write_dir,
        num_depth_bins=motion_num_depth_bins,
        training_steps=2000,
        **medicine_kwargs,
    )

    print("\n\nDONE\n\n")

    return trainer
