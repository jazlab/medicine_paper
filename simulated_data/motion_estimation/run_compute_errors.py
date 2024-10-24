"""Compute motion estimation errors.

See ../README.md for more information and usage instructions.
"""

import json
import sys
from pathlib import Path

import numpy as np
import utils
from matplotlib import pyplot as plt
from scipy import interpolate as scipy_interpolate

_N_DEPTH_LEVELS = 11


def compute_true_drift_values(
    eval_depths: np.ndarray,
    true_drift: np.ndarray,
    unit_depths: np.ndarray,
    external_drift_factors: np.ndarray,
) -> np.ndarray:
    """Compute true drift values for each evaluation depth.

    Args:
        eval_depths: np.ndarray, shape (n_timesteps, n_depth_levels).
        true_drift: np.ndarray, shape (n_timesteps,).
        unit_depths: np.ndarray, shape (n_units,).
        external_drift_factors: np.ndarray, shape (n_units,).

    Returns:
        true_drift: np.ndarray, shape (n_timesteps, n_depth_levels).
    """
    # Find linear mapping from depth to drift scale
    min_unit_ind = np.argmin(unit_depths)
    max_unit_ind = np.argmax(unit_depths)
    min_unit_depth = unit_depths[min_unit_ind]
    max_unit_depth = unit_depths[max_unit_ind]
    min_drift_factor = external_drift_factors[min_unit_ind]
    max_drift_factor = external_drift_factors[max_unit_ind]
    slope = (max_drift_factor - min_drift_factor) / (
        max_unit_depth - min_unit_depth
    )
    intercept = min_drift_factor - min_unit_depth * slope

    # Compute ground truth drift values
    # eval_depth = orig_depth + drift_factor(depth) * true_drift
    #     = orig_depth + (slope * orig_depth + intercept) * true_drift
    #     = (1 + slope * true_drift) * orig_depth + intercept * true_drift
    # orig_depth = (
    #     (eval_depth - intercept * true_drift) /
    #     (1 + slope * true_drift)
    # )
    eval_depths = eval_depths[None] * np.ones((len(true_drift), 1))
    orig_depths = (eval_depths - intercept * true_drift[:, None]) / (
        1 + slope * true_drift[:, None]
    )
    true_drift = eval_depths - orig_depths

    return true_drift


def compute_error(estimator_name: str, recording_dir: Path) -> None:
    """Compute error for estimator on recording directory.

    Args:
        estimator_name: Name of motion estimator.
        recording_dir: Path to recording directory.
    """
    dataset_name = recording_dir.name
    print(f"\nOn {dataset_name} running {estimator_name}")
    motion_estimation_dir = (
        utils.MOTION_ESTIMATION_DIR / dataset_name / estimator_name
    )

    # Continue if already ran
    if (motion_estimation_dir / "errors_l1.npy").exists():
        return

    # Find min and max depth
    min_eval_depth = -np.inf
    max_eval_depth = np.inf
    for estimator_dir in motion_estimation_dir.parent.iterdir():
        if estimator_dir.name[0] == ".":
            continue
        depth_bins = np.load(estimator_dir / "depth_bins.npy")
        if len(depth_bins) == 1:
            continue
        min_eval_depth = max(min_eval_depth, np.min(depth_bins))
        max_eval_depth = min(max_eval_depth, np.max(depth_bins))

    # Load drift times and values
    drift_dict = np.load(
        recording_dir / "drift_dict.npy", allow_pickle=True
    ).item()
    eval_times = drift_dict["external_drift_times"]
    eval_depths = np.linspace(min_eval_depth, max_eval_depth, _N_DEPTH_LEVELS)
    true_drift_vector = drift_dict["external_drift_vector_um"]
    external_drift_factors = drift_dict["external_drift_factors"]

    # Remove first and last 50 values to ensure interpolation is valid
    eval_times = eval_times[50:-50]
    true_drift_vector = true_drift_vector[50:-50]

    # Load unit depths
    unit_depths = np.load(recording_dir / "unit_depths.npy")

    # Compute true drift values
    true_drift_values = compute_true_drift_values(
        eval_depths, true_drift_vector, unit_depths, external_drift_factors
    )

    # Load motion array
    pred_motion = np.load(motion_estimation_dir / "motion.npy")
    if len(pred_motion.shape) == 3:
        if pred_motion.shape[0] == 1:
            pred_motion = pred_motion[0]
        elif pred_motion.shape[1] == 1:
            pred_motion = pred_motion[:, 0]
        else:
            raise ValueError(f"Unknown motion shape {pred_motion.shape}")

    # Load time bins and depth bins
    pred_times = np.load(motion_estimation_dir / "time_bins.npy")
    if len(pred_times.shape) == 2 and pred_times.shape[0] == 1:
        pred_times = pred_times[0]
    pred_depths = np.load(motion_estimation_dir / "depth_bins.npy")

    # For rigid motion correction, must expand depth
    if len(pred_depths) == 1:
        pred_depths = np.linspace(
            min_eval_depth, max_eval_depth, _N_DEPTH_LEVELS
        )
        pred_motion = np.repeat(pred_motion, _N_DEPTH_LEVELS, axis=1)

    # Compute interpolation
    pred_times_depths = (pred_times, pred_depths)
    eval_times_depths_grid = np.stack(
        [x.T for x in np.meshgrid(eval_times, eval_depths)], axis=2
    )
    eval_times_depths_grid_flat = eval_times_depths_grid.reshape(-1, 2)
    f = scipy_interpolate.RegularGridInterpolator(
        pred_times_depths,
        pred_motion,
        method="linear",
        bounds_error=False,
        fill_value=np.nan,
    )
    pred_drift_values_flat = f(eval_times_depths_grid_flat)
    pred_drift_values = pred_drift_values_flat.reshape(
        eval_times_depths_grid.shape[:2]
    )

    # Compute L1 error vectors
    pred_drift_values_median = np.median(
        pred_drift_values, axis=0, keepdims=True
    )
    true_drift_values_median = np.median(
        true_drift_values, axis=0, keepdims=True
    )
    pred_drift_values_no_median = pred_drift_values - pred_drift_values_median
    true_drift_values_no_median = true_drift_values - true_drift_values_median
    errors_l1 = np.abs(
        true_drift_values_no_median - pred_drift_values_no_median
    )

    # Save errors
    print(f"Saving to {motion_estimation_dir}")
    motion_estimation_dir.mkdir(parents=True, exist_ok=True)
    np.save(motion_estimation_dir / "errors_l1.npy", errors_l1)
    np.save(motion_estimation_dir / "mean_error_l1.npy", np.mean(errors_l1))

    # Plot true drift and predicted drift
    fig, ax = plt.subplots(figsize=(3, 6))
    ax.plot(
        eval_times,
        eval_depths + true_drift_values_no_median,
        label="True drift",
        c="g",
    )
    ax.plot(
        eval_times,
        eval_depths + pred_drift_values_no_median,
        label="Pred drift",
        c="r",
    )
    ax.set_xlabel("Time (sec)")
    ax.set_ylabel("Deviation ($\mu$m)")

    # Save figure
    fig.savefig(motion_estimation_dir / f"drift_plot.png")
    plt.close(fig)


def main(
    estimator_names: None | str = None, **recording_hyperparameters: dict
):
    """Compute errors for motion estimators on recordings.

    Args:
        estimator_names: None or comma-separated string of estimator names. If
            None, run all estimators.
        recording_hyperparameters: Dict of hyperparameters to filter the
            datasets. Keys must be: drift_rate, random_walk_sigma,
            random_jump_rate, unit_num, homogeneity, stability, non_rigidity.
    """
    utils.run_over_estimators_and_datasets(
        compute_error,
        estimator_names=estimator_names,
        **recording_hyperparameters,
    )
    print("\n\nDONE\n\n")


if __name__ == "__main__":
    main(**dict(arg.split("=") for arg in sys.argv[1:]))
