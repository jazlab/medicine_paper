"""Functions for loading sorting results data.

For usage, see ../../figures/spike_sorting_results.ipynb.
"""

from pathlib import Path

import numpy as np
import pandas as pd


def load_sorting_results(sorting_dir: Path) -> tuple:
    """Load spike sorting results.

    Args:
        sorting_dir: Path to spike sorting results directory.

    Returns:
        spike_times_per_cluster: list of np.ndarray, spike times per cluster.
        spike_depths_per_cluster: list of np.ndarray, spike depths per cluster.
        labels: list of str, cluster labels.
    """
    # Read kilosort outputs
    ks_outputs = {}
    ks_outputs["spike_times"] = np.squeeze(
        np.load(sorting_dir / "spike_times.npy") / 32000
    )
    ks_outputs["spike_clusters"] = np.squeeze(
        np.load(sorting_dir / "spike_clusters.npy")
    )
    ks_outputs["templates_ind"] = np.load(sorting_dir / "templates_ind.npy")
    ks_outputs["templates"] = np.load(sorting_dir / "templates.npy")
    ks_outputs["spike_templates"] = np.squeeze(
        np.load(sorting_dir / "spike_templates.npy")
    )
    ks_outputs["channel_positions"] = np.load(
        sorting_dir / "channel_positions.npy"
    )
    ks_outputs["channel_map"] = np.squeeze(
        np.load(sorting_dir / "channel_map.npy")
    )
    ks_outputs["spike_positions"] = np.load(
        sorting_dir / "spike_positions.npy"
    )

    # Extract spike depths
    ks_outputs["spike_depths"] = ks_outputs["spike_positions"][:, 1]

    # Get spike times per cluster
    unique_cluster_ids = sorted(np.unique(ks_outputs["spike_clusters"]))
    unique_cluster_ids = [int(x) for x in unique_cluster_ids]
    spike_times_per_cluster = [
        ks_outputs["spike_times"][ks_outputs["spike_clusters"] == cluster_id]
        for cluster_id in unique_cluster_ids
    ]
    # Get spike depths per cluster
    spike_depths_per_cluster = [
        ks_outputs["spike_depths"][ks_outputs["spike_clusters"] == cluster_id]
        for cluster_id in unique_cluster_ids
    ]

    # Read cluster labels
    labels_original = np.array(
        pd.read_csv(sorting_dir / "cluster_KSLabel.tsv", sep="\t").values
    )
    labels = []
    for x in unique_cluster_ids:
        index = np.argwhere(labels_original[:, 0] == x)[0, 0]
        labels.append(labels_original[index, 1])

    return spike_times_per_cluster, spike_depths_per_cluster, labels


def load_ground_truth_results(
    ground_truth_dir: Path,
    dataset_dir: Path,
) -> tuple:
    """Load ground truth results.

    Args:
        ground_truth_dir: Path to ground truth directory.
        dataset_dir: Path to dataset directory.

    Returns:
        true_spike_trains: list of np.ndarray, true spike trains.
        true_depths_mean: np.ndarray, mean true depths.
        drift_dict: dict, drift values.
    """
    # Load ground truth spike trains
    true_spike_trains = [
        np.load(x) for x in (ground_truth_dir / "spike_trains").iterdir()
    ]
    true_depths = np.load(ground_truth_dir / "template_depths.npy")
    drift_dict = np.load(
        dataset_dir / "drift_dict.npy", allow_pickle=True
    ).item()
    true_depths_mean = np.mean(true_depths, axis=1)

    return true_spike_trains, true_depths_mean, drift_dict
