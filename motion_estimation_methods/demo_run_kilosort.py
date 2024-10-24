"""Demo run medicine."""

from pathlib import Path

import run_kilosort


def main():
    write_dir = Path("./tmp_ks")
    write_dir.mkdir(exist_ok=True, parents=True)
    run_kilosort.run_kilosort(
        recording_dir=Path("../cache/simulated_datasets/0272"),
        write_dir=write_dir,
        random_seed=0,
    )


if __name__ == "__main__":
    main()
