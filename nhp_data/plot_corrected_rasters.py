"""Plot corrected rasters."""

import sys
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

medicine_path = (
    "/Users/nicholaswatters/Desktop/grad_school/research/mehrdad/medicine"
)
sys.path.append(medicine_path)
from medicine import plotting as medicine_plotting

_RESULTS_DIR = Path("../../cache/monkey_data/peaks")
_CACHE_FIGURE_DIR = Path("../../cache/monkey_data/motion_correction_figures")

_SAMPLING_FREQUENCY = 30000


def _plot_corrected_rasters(
    estimator_dir, peak_times, peak_depths, peak_amplitudes
):
    print(f"Plotting corrected rasters for {estimator_dir}")

    # Load motion array
    motion_path = estimator_dir / "motion.npy"
    if not motion_path.exists():
        print(f"No motion.npy found for {estimator_dir}")
        return
    pred_motion = np.load(estimator_dir / "motion.npy")

    # Load time bins
    if (estimator_dir / "temporal_bins.npy").exists():
        pred_times = np.load(estimator_dir / "temporal_bins.npy")
    else:
        pred_times = np.load(estimator_dir / "time_bins.npy")

    if "medicine" in estimator_dir.name:
        pred_times *= 32000 / 30000

    # Load depth bins
    if (estimator_dir / "spatial_bins.npy").exists():
        pred_depths = np.load(estimator_dir / "spatial_bins.npy")
    else:
        pred_depths = np.load(estimator_dir / "depth_bins.npy")

    if estimator_dir.name == "kilosort_like":
        pred_depths = pred_depths[::5]
        pred_motion = pred_motion[:, ::5]
    else:
        return

    # Make figure
    fig = medicine_plotting.plot_motion_correction(
        peak_times=peak_times,
        peak_depths=peak_depths,
        peak_amplitudes=peak_amplitudes,
        time_bins=pred_times,
        depth_bins=pred_depths,
        motion=pred_motion,
    )

    # Save figure
    fig_dir = _CACHE_FIGURE_DIR / estimator_dir.parent.parent.name
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_dir / f"{estimator_dir.name}.pdf")
    plt.close(fig)


def main():
    """Plot corrected rasters."""

    for results_dir in _RESULTS_DIR.iterdir():
        if results_dir.name[0] == ".":
            continue

        # if results_dir.name != 'Perle_2022-06-06':
        #     continue

        # Load times and amplitudes
        peaks = np.load(results_dir / "peaks.npy")
        peaks = np.array([x.tolist() for x in peaks])
        peaks = dict(
            sample_index=peaks[:, 0],
            channel_index=peaks[:, 1],
            amplitude=peaks[:, 2],
            segment_index=peaks[:, 3],
        )
        peak_amplitudes = peaks["amplitude"]
        peak_times = peaks["sample_index"] / _SAMPLING_FREQUENCY

        # Load peak depths
        peak_locations = np.load(results_dir / "peak_locations.npy")
        peak_locations = np.array([x.tolist() for x in peak_locations])
        peak_locations = dict(
            x=peak_locations[:, 0],
            y=peak_locations[:, 1],
            z=peak_locations[:, 2],
            alpha=peak_locations[:, 3],
        )
        peak_depths = peak_locations["y"]

        for estimator_dir in (results_dir / "motion_estimation").iterdir():
            # Make plot
            _plot_corrected_rasters(
                estimator_dir=estimator_dir,
                peak_times=peak_times,
                peak_depths=peak_depths,
                peak_amplitudes=peak_amplitudes,
            )

    print("\n\nDONE\n\n")


if __name__ == "__main__":
    main()
