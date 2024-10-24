"""Get a set of 40 datasets.

This script gets a set of 40 datasets from the 384 datasets available. These are
the datasets used for the spike sorting methods in the MEDiCINe paper.
"""

import numpy as np


def get_dataset_names() -> list[str]:
    """Get set of 40 datasets.

    Returns:
        list: List of dataset names.
    """
    np.random.seed(0)
    dataset_numbers = np.unique(np.random.randint(1, 385, 45))
    print(f"Number of datasets: {len(dataset_numbers)}")
    dataset_names = [str(x).zfill(4) for x in dataset_numbers]
    print(f"Datasets:\n{dataset_names}")

    return dataset_names


if __name__ == "__main__":
    get_dataset_names()
