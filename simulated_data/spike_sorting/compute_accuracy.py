"""Compute spike sorting accuracy.

This compute accuracy statistics for spike sorting. It is used to evaluate the
spike sorting quality after different motion correction methods.

For usage, see ../../figures/spike_sorting_results.ipynb.
"""

import numpy as np


def _train_to_bool(
    train_indices: np.ndarray, smooth: bool, max_spike_index: int
) -> np.ndarray:
    """Convert spike train to boolean array.

    Args:
        train_indices: np.ndarray, shape (n_spikes,). Integer indices of spikes
            in the spike train, after discretization by a time bin.
        smooth: bool, whether to smooth spike train. If True, the each spike
            will be smoothed by adding spikes at the previous and next
            timesteps.
        max_spike_index: int, maximum spike index.

    Returns:
        train_bool: np.ndarray, shape (max_spike_index,). Boolean array of
            spikes in the spike train.
    """
    train_bool = np.zeros(max_spike_index, dtype=bool)
    train_bool[train_indices] = True
    if smooth:
        indices_plus_1 = train_indices + 1
        indices_plus_1 = indices_plus_1[indices_plus_1 < max_spike_index]
        train_bool[indices_plus_1] = True
        indices_minus_1 = train_indices - 1
        indices_minus_1 = indices_minus_1[indices_minus_1 >= 0]
        train_bool[indices_minus_1] = True
    return train_bool


def compute_accuracy_statistics(
    true_spike_trains: list[np.ndarray],
    sorting_spike_trains: list[np.ndarray],
    threshold_ms: float = 3.0,
) -> np.ndarray:
    """Compute merging statistics to get true from sorting spike trains.

    Args:
        true_spike_trains: list of np.ndarray, shape (n_spikes,). List of true
            spike trains.
        sorting_spike_trains: list of np.ndarray, shape (n_spikes,). List of
            sorting spike trains.
        threshold_ms: float, threshold in milliseconds for merging spikes.

    Returns:
        accuracy_statistics: np.ndarray, shape (n_true_units, n_sorting_units).
            Array of accuracy statistics for each true unit and each sorting
            unit.
    """
    threshold_sec = threshold_ms / 1000
    num_true_units = len(true_spike_trains)

    # Check that true spike trains are non-negative
    smallest_true_spike = min([min(x) for x in true_spike_trains])
    if smallest_true_spike < 0:
        raise ValueError("True spike trains must be non-negative.")

    # Compute maximum spike index
    max_true_spike = max([max(x) for x in true_spike_trains])
    max_sorting_spike = max([max(x) for x in sorting_spike_trains])
    max_spike = max(max_true_spike, max_sorting_spike)
    max_spike_index = int(max_spike / threshold_sec) + 2

    # Compute spike train indices
    true_train_indices = [
        np.round(train / threshold_sec).astype(int)
        for train in true_spike_trains
    ]
    sorting_train_indices = [
        np.round(train / threshold_sec).astype(int)
        for train in sorting_spike_trains
    ]

    # Compute boolean sorting spike trains
    sorting_trains_smooth = [
        _train_to_bool(x, smooth=True, max_spike_index=max_spike_index)
        for x in sorting_train_indices
    ]

    # Iterate through true spike trains and compute accuracy statistics for it
    accuracy_statistics = []
    for i in range(num_true_units):
        # Compute coverage masks for each sorting unit
        true_train_inds = true_train_indices[i]
        num_true_spikes = len(true_train_inds)

        # Get the accuracy for each sorting unit
        accuracy_per_sorting_unit = []
        for sorting_inds, sorting_smooth in zip(
            sorting_train_indices,
            sorting_trains_smooth,
        ):
            overlap = np.sum(sorting_smooth[true_train_inds])
            num_sorting_spikes = len(sorting_inds)
            false_positive = (
                num_sorting_spikes - overlap
            ) / num_sorting_spikes
            false_negative = (num_true_spikes - overlap) / num_true_spikes
            accuracy = 1 - false_negative - false_positive
            accuracy_per_sorting_unit.append(accuracy)
        accuracy_statistics.append(accuracy_per_sorting_unit)
    accuracy_statistics = np.array(accuracy_statistics)

    return accuracy_statistics
