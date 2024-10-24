"""Run motion estimators on simulated datasets.

See ../README.md for more information and usage instructions.
"""

import json
from pathlib import Path

# Get true path of this file
_CURRENT_PATH = Path(__file__).resolve().parent
SIMULATED_DATASETS_DIR = _CURRENT_PATH / "../../cache/simulated_datasets"
MOTION_ESTIMATION_DIR = _CURRENT_PATH / "../../cache/sim_motion_estimation"
_ESTIMATOR_NAMES = [
    "kilosort_datashift",
    "dredge",
    "rigid_fast",
    "medicine_rigid",
    "medicine",
]


def get_recording_dirs(**recording_hyperparameters: dict) -> list:
    """Check if dataset with given hyperparameters already exists.

    Args:
        recording_hyperparameters: Dict of hyperparameters. Keys must be:
            drift_rate, random_walk_sigma, random_jump_rate, unit_num,
            homogeneity, stability, non_rigidity.

    Returns:
        list: List of recording directories that match the given
            hyperparameters.
    """
    recording_dirs = []
    for dataset_dir in SIMULATED_DATASETS_DIR.iterdir():
        # Fetch hyperparameters from dataset_dir
        dataset_hyperparams_path = dataset_dir / "hyperparameters.json"
        if not dataset_hyperparams_path.is_file():
            continue
        dataset_hyperparams = json.load(open(dataset_hyperparams_path, "r"))

        # Sanity check to make sure the keys are the same
        recording_keys = set(recording_hyperparameters.keys())
        dataset_keys = set(dataset_hyperparams.keys())
        if not recording_keys.issubset(dataset_keys):
            raise ValueError(
                f"Inconsistent hyperparameters for dataset {dataset_dir}: "
                f"{dataset_keys} does not contain {recording_keys}"
            )

        # Check if any hyperparameters do not match
        match = True
        for key, value in recording_hyperparameters.items():
            if dataset_hyperparams[key] != value:
                match = False
                break

        # If all hyperparameters match, return True
        if match:
            recording_dirs.append(dataset_dir)

    return recording_dirs


def curate_hyperparams(recording_hyperparameters: dict) -> dict:
    """Curate hyperparameters.

    Args:
        recording_hyperparameters: Dict of hyperparameters. Keys must be:
            drift_rate, random_walk_sigma, random_jump_rate, unit_num,
            homogeneity, stability, non_rigidity.

    Returns:
        dict: Curated hyperparameters with correct types.
    """
    float_keys = [
        "drift_rate",
        "random_walk_sigma",
        "random_jump_rate",
        "non_rigidity",
    ]
    int_keys = ["unit_num"]
    str_keys = ["homogeneity", "stability"]
    curated_hyperparams = {}
    for key, value in recording_hyperparameters.items():
        if key in float_keys:
            curated_hyperparams[key] = float(value)
        elif key in int_keys:
            curated_hyperparams[key] = int(value)
        elif key in str_keys:
            curated_hyperparams[key] = str(value)
        else:
            raise ValueError(f"Unknown recording hyperparameter {key}")
    return curated_hyperparams


def run_over_estimators_and_datasets(
    f: callable,
    estimator_names: None | str = None,
    **recording_hyperparameters: dict,
):
    """Run function over estimators and datasets.

    This function allows you to run a function over multiple estimators and
    datasets.

    Args:
        f: Function taking in estimator_name, recording_dir. This is evaluated
            for each estimator_name and recording_dir.
        estimator_names: None or comma-separated string of estimator names. If
            None, run all estimators.
        recording_hyperparameters: Dict of hyperparameters to filter the
            datasets. Keys must be: drift_rate, random_walk_sigma,
            random_jump_rate, unit_num, homogeneity, stability, non_rigidity.
    """
    # Curate hyperparameters
    recording_hyperparameters = curate_hyperparams(recording_hyperparameters)

    # Find recording paths
    recording_dirs = get_recording_dirs(**recording_hyperparameters)

    # Get estimators to run
    if estimator_names is None:
        estimator_names = _ESTIMATOR_NAMES
    else:
        estimator_names = estimator_names.split(",")

    # Run estimator(s) on all recording directories
    for recording_dir in recording_dirs:
        # Create recording directory
        recording_dir = SIMULATED_DATASETS_DIR / recording_dir.name

        # Run function on each estimator for this recording
        for estimator_name in estimator_names:
            f(estimator_name, recording_dir)
