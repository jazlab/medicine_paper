"""Demo run medicine."""

from pathlib import Path

import run_medicine


def main():
    write_dir = Path("./tmp_medicine")
    write_dir.mkdir(exist_ok=True, parents=True)
    run_medicine.run_medicine(
        recording_dir=Path("../cache/simulated_data/datasets/0001"),
        write_dir=write_dir,
        training_steps=20,
        sampling_frequency=32000,
        num_depth_bins=20,
        motion_num_depth_bins=2,
        motion_bound=800,
        random_seed=0,
    )


if __name__ == "__main__":
    main()
