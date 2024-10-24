"""Run motion estimators on simulated datasets.

See ../README.md for more information and usage instructions.
"""

import sys
from pathlib import Path

import utils

sys.path.append("../../motion_estimation_methods")
import run_kilosort
import run_medicine
import run_spikeinterface


def run_motion_estimator(estimator_name: str, recording_dir: Path) -> None:
    """Run motion estimator on recording directory.

    Args:
        estimator_name: Name of motion estimator.
        recording_dir: Path to recording directory.
    """
    dataset_name = recording_dir.name
    print(f"\nOn {dataset_name} running {estimator_name}")

    # Make motion_estimation_dir
    motion_estimation_dir = (
        utils.MOTION_ESTIMATION_DIR / dataset_name / estimator_name
    )
    motion_estimation_dir.mkdir(exist_ok=True, parents=True)

    # Continue if already ran
    if (motion_estimation_dir / "motion.npy").exists():
        print(f"Already ran {estimator_name} on {recording_dir}")
        return

    # Run estimator on recording
    if estimator_name == "medicine":
        run_medicine.run_medicine(
            recording_dir=recording_dir,
            write_dir=motion_estimation_dir,
        )
    elif estimator_name == "medicine_rigid":
        run_medicine.run_medicine(
            recording_dir=recording_dir,
            write_dir=motion_estimation_dir,
            motion_num_depth_bins=1,
        )
    elif estimator_name == "kilosort_datashift":
        run_kilosort.run_kilosort(
            recording_dir=recording_dir,
            write_dir=motion_estimation_dir,
        )
    else:
        run_spikeinterface.run_spikeinterface_from_mearec(
            recording_dir=recording_dir,
            write_dir=motion_estimation_dir,
            estimator_name=estimator_name,
        )


def main(
    estimator_names: None | str = None, **recording_hyperparameters: dict
):
    """Run motion estimators on recordings.

    Args:
        estimator_names: None or comma-separated string of estimator names. If
            None, run all estimators.
        recording_hyperparameters: Dict of hyperparameters to filter the
            datasets. Keys must be: drift_rate, random_walk_sigma,
            random_jump_rate, unit_num, homogeneity, stability, non_rigidity.
    """
    utils.run_over_estimators_and_datasets(
        run_motion_estimator,
        estimator_names=estimator_names,
        **recording_hyperparameters,
    )
    print("\n\nDONE\n\n")


if __name__ == "__main__":
    main(**dict(arg.split("=") for arg in sys.argv[1:]))
