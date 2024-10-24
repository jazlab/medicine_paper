"""Extract and save peaks.

The main entry point for the script is `extract_and_plot_peaks`. This function
loads a recording, runs peak detection, saves the peaks and peak locations, and
plots the peaks. See ./run_generate_data.py for an example usage.
"""

import numpy as np
from matplotlib import pyplot as plt
from spikeinterface.core.node_pipeline import (
    ExtractDenseWaveforms,
    run_node_pipeline,
)
from spikeinterface.extractors import read_mearec
from spikeinterface.sortingcomponents.peak_detection import detect_peak_methods
from spikeinterface.sortingcomponents.peak_localization import (
    localize_peak_methods,
)

_LOCALIZE_METHOD = "monopolar_triangulation"
_LOCALIZE_KWARGS = dict(
    radius_um=75.0,
    max_distance_um=150.0,
    optimizer="minimize_with_log_penality",
    enforce_decrease=True,
    feature="ptp",
)
_DETECT_METHOD = "locally_exclusive"
_DETECT_KWARGS = dict(
    detect_threshold=10,
)
_JOB_KWARGS = dict(
    chunk_duration="1s",
    n_jobs=-1,
    progress_bar=True,
)


def run_peak_detection(recording) -> tuple[dict, dict]:
    """Extract peaks and locations."""
    detect_method_class = detect_peak_methods[_DETECT_METHOD]
    node0 = detect_method_class(recording, **_DETECT_KWARGS)
    node1 = ExtractDenseWaveforms(
        recording, parents=[node0], ms_before=0.1, ms_after=0.3
    )
    localize_peaks_method_class = localize_peak_methods[_LOCALIZE_METHOD]
    node2 = localize_peaks_method_class(
        recording,
        parents=[node0, node1],
        return_output=True,
        **_LOCALIZE_KWARGS,
    )
    pipeline_nodes = [node0, node1, node2]
    peaks, peak_locations = run_node_pipeline(
        recording,
        pipeline_nodes,
        _JOB_KWARGS,
        job_name="detect and localize",
        gather_mode="memory",
        gather_kwargs=None,
        squeeze_output=False,
        folder=None,
        names=None,
    )

    return peaks, peak_locations


def plot_peaks(
    peak_times,
    peak_locations,
    peak_amplitudes,
    stride=3,
    alpha=0.75,
    colormap="winter",
):
    """Plot raster-map of peaks."""
    # Subsample
    peak_times = peak_times[::stride]
    peak_locations = peak_locations[::stride]
    peak_amplitudes = peak_amplitudes[::stride]

    # Normalize amplitudes by CDF to have uniform distribution
    amp_argsort = np.argsort(np.argsort(peak_amplitudes))
    peak_amplitudes = amp_argsort / len(peak_amplitudes)

    # Get colors and create figure
    cmap = plt.get_cmap(colormap)
    colors = cmap(peak_amplitudes)
    fig, ax = plt.subplots(1, 1, figsize=(5, 10))
    plot = ax.scatter(peak_times, peak_locations, s=1, c=colors, alpha=alpha)
    ax.set_xlabel("time (s)", fontsize=12)
    ax.set_ylabel("depth from probe tip ($\mu$m)", fontsize=12)
    fig.colorbar(plot, ax=ax)

    return fig


def extract_and_plot_peaks(recording_dir, sampling_frequency=32000):
    """Extract, save, and plot peaks."""
    print(f"\nRunning peak extraction and plotting for {recording_dir}")

    # Load recording and run peak detection
    raw_recording, _ = read_mearec(recording_dir / "recording.h5")
    peaks, peak_locations = run_peak_detection(raw_recording)

    # Save peaks and peak locations to write_dir
    print(f"Saving peaks and peak_locations to {recording_dir}")
    np.save(recording_dir / "peaks.npy", peaks)
    np.save(recording_dir / "peak_locations.npy", peak_locations)

    # Plot peaks
    fig_path = recording_dir / f"peaks.png"
    print(f"Plotting peaks and saving to {fig_path}")
    peak_locations_y = peak_locations["y"]
    peak_amplitudes = peaks["amplitude"]
    peak_times = peaks["sample_index"] / sampling_frequency
    fig = plot_peaks(peak_times, peak_locations_y, peak_amplitudes)
    fig.savefig(fig_path)
