"""Generate simulated data.

Usage:
```
python run_generate_data.py
```

You may constrain the hyperparameters by passing them as arguments. For example:
```
python run_generate_data.py drift_rates=0.1 random_walk_sigmas=1,2
```
This will generate datasets for only drift rates of 0.1 um/s and random walk
sigmas of 1 and 2 um/sqrt(s).

Valid arguments are:
- drift_rates: Drift rates in um/s.
- random_walk_sigmas: Standard deviations of random walks in um/sqrt(s).
- random_jump_rates: Rates of Poisson random jumps in 1/s.
- unit_nums: Number of units.
- homogeneities: Homogeneity of units.
- stabilities: Stability of units.
- non_rigidities: Non-rigidity of units. 0 means rigid (uniform) motion across
    depth.
See the `main()` function for the default values.

The generated datasets will be saved in `../../cache/simulated_datasets`,
along with the drift vectors, extracted peaks, and a couple summary plots.
"""

import collections
import itertools
import json
import sys
from pathlib import Path

import drift_lib
import elephant
import MEArec
import MEAutility
import neo
import numpy as np
import peak_extraction
import quantities as pq
from matplotlib import pyplot as plt

_BASE_DATA_DIR = Path("../../cache")
_TEMPLATE_DIR = _BASE_DATA_DIR / "simulated_neuron_templates"
_SIMULATED_DATASETS_DIR = _BASE_DATA_DIR / "simulated_datasets"
_TMP_DIR = _BASE_DATA_DIR / "tmp"
_SIMULATED_DATASETS_DIR.mkdir(parents=True, exist_ok=True)


def _data_exists(hyperparameters: dict) -> bool:
    """Check if dataset with given hyperparameters already exists.

    Args:
        hyperparameters: Dict of hyperparameters. Keys must be:
            drift_rate, random_walk_sigma, random_jump_rate, unit_num,
            homogeneity, stability, non_rigidity.

    Returns:
        bool: True if dataset with given hyperparameters exists, False
            otherwise.
    """
    for dataset_dir in _SIMULATED_DATASETS_DIR.iterdir():
        # Fetch hyperparameters from dataset_dir
        dataset_hyperparams_path = dataset_dir / "hyperparameters.json"
        if not dataset_hyperparams_path.is_file():
            continue
        dataset_hyperparams = json.load(open(dataset_hyperparams_path, "r"))

        # Sanity check to make sure the keys are the same
        if set(dataset_hyperparams.keys()) != set(hyperparameters.keys()):
            raise ValueError(
                f"Inconsistent hyperparameters for dataset {dataset_dir}: "
                f"{dataset_hyperparams}, but expected {hyperparameters}"
            )

        # Check if any hyperparameters do not match
        match = True
        for key, value in hyperparameters.items():
            if dataset_hyperparams[key] != value:
                match = False
                break

        # If all hyperparameters match, return True
        if match:
            return True

    # If no matching dataset was found, return False
    return False


def _get_templates(
    probename: str = "Neuropixels-128",
    drift_limits: tuple[float, float] = (-200, 200),
) -> MEArec.TemplateGenerator:
    """Generate templates.

    Args:
        probename: Probe name.
        drift_limits: Drift limits in um.

    Returns:
        MEArec.TemplateGenerator: Template generator.
    """
    # Check if templates already exist
    template_filename = _TEMPLATE_DIR / f"templates_{probename}.h5"
    if template_filename.is_file():
        print(f"{template_filename} already generated")
        tempgen = MEArec.load_templates(template_filename)
        return tempgen

    # Get template parameters
    template_params = MEArec.get_default_templates_params()
    template_params["probe"] = probename
    template_params["n"] = 50  # Number of templates per cell model
    template_params["drifting"] = True
    # Number of drift steps. Here we define as number of microns. Drift will be
    # discretized into this number of steps.
    template_params["drift_steps"] = drift_limits[1] - drift_limits[0]
    template_params["drift_xlim"] = [0, 0]
    template_params["drift_ylim"] = [0, 0]
    template_params["drift_zlim"] = [
        drift_limits[1] - drift_limits[0],
        drift_limits[1] - drift_limits[0],
    ]
    template_params["max_drift"] = drift_limits[1] - drift_limits[0]
    template_params["min_drift"] = 0
    electrode_positions = MEAutility.return_mea(probename).positions
    template_params["zlim"] = [
        electrode_positions[:, 2].min(),
        electrode_positions[:, 2].max(),
    ]
    template_params["overhang"] = 0
    template_params["min_amp"] = 0

    # Generate templates
    print(f"Generating templates in {template_filename}")
    cell_folder = MEArec.get_default_cell_models_folder()
    tempgen = MEArec.gen_templates(
        cell_models_folder=cell_folder,
        params=template_params,
        templates_tmp_folder=_TMP_DIR / "MEArec_templates",
        verbose=True,
        recompile=True,
    )
    MEArec.save_template_generator(tempgen, filename=template_filename)
    tempgen = MEArec.load_templates(template_filename)

    return tempgen


def _select_template_indices(
    tempgen: MEArec.TemplateGenerator,
    homogeneity: str,
    unit_num: int,
) -> np.ndarray:
    """Select template indices.

    Args:
        tempgen: Template generator.
        homogeneity: Homogeneity of units. Must be 'uniform' or 'bimodal'.
        unit_num: Number of units.

    Returns:
        templates_ids: Indices of the units of the template to use.
    """
    # Get valid depths
    ptps = tempgen.templates[:].ptp(axis=(1, 2, 3))
    possible_inds = np.argwhere((ptps > 5) & (ptps < 300))[:, 0]
    drift_steps = tempgen.locations.shape[1]
    center_depths = tempgen.locations[:, drift_steps // 2, 2]
    valid_depths = center_depths[possible_inds]

    # Sample template indices
    if homogeneity == "uniform":
        templates_ids = np.random.choice(
            possible_inds, unit_num, replace=False
        )
    elif homogeneity == "bimodal":
        # Create bimodal gaussian distribution
        electrode_zlim = tempgen.params["params"]["zlim"]
        probe_length = electrode_zlim[1] - electrode_zlim[0]
        mu_0 = 0.15 * probe_length + electrode_zlim[0]
        mu_1 = 0.85 * probe_length + electrode_zlim[0]
        sigma = probe_length / 10
        distrib = np.exp(
            -((valid_depths - mu_0) ** 2) / (2 * (sigma**2))
        ) + np.exp(-((valid_depths - mu_1) ** 2) / (2 * (sigma**2)))
        distrib /= np.sum(distrib)
        templates_ids = np.random.choice(
            possible_inds,
            unit_num,
            p=distrib,
            replace=False,
        )
    else:
        raise ValueError(f"Invalid homogeneity: {homogeneity}")

    return templates_ids


def _get_recording_params(
    duration: float,
    unit_num: int,
    random_seed: int,
) -> dict:
    """Get recording parameters.

    Args:
        duration: Duration of recording in seconds.
        unit_num: Number of units.
        random_seed: Random seed.

    Returns:
        dict: Recording parameters for MEArec.gen_recordings().
    """
    recording_params = MEArec.get_default_recordings_params()

    # Number of excitatory and inhibitory units
    n_exc = int(0.75 * unit_num)
    n_inh = unit_num - n_exc

    # Revisit cell type numbers
    print(f"n_exc: {n_exc}, n_inh: {n_inh}")

    # Set recording parameters
    recording_params["spiketrains"]["duration"] = duration
    recording_params["spiketrains"]["f_exc"] = 5
    recording_params["spiketrains"]["f_inh"] = 5
    recording_params["templates"]["min_amp"] = 30
    recording_params["templates"]["min_dist"] = 20
    recording_params["templates"]["smooth_percent"] = 0
    recording_params["templates"]["n_jitters"] = 3
    recording_params["recordings"]["filter"] = False
    recording_params["recordings"]["noise_level"] = 5
    recording_params["recordings"]["noise_mode"] = "distance-correlated"
    recording_params["recordings"]["chunk_duration"] = 1.0
    recording_params["recordings"]["modulation"] = "template"
    recording_params["recordings"]["bursting"] = False
    recording_params["recordings"]["drifting"] = True

    # Add random seeds for reproducibility if necessary
    recording_params["seeds"]["spiketrains"] = random_seed
    recording_params["seeds"]["templates"] = random_seed
    recording_params["seeds"]["convolution"] = random_seed
    recording_params["seeds"]["noise"] = random_seed

    return recording_params


def _get_firing_rate_periodic(
    duration: float,
    phase: None | float = None,
    period: float = 4 * 60,
    max_fr: float = 5,
    min_fr_before_clipping: float = -2,
    min_fr: float = 0.2,
) -> np.ndarray:
    """Get periodic firing rate vector.

    Args:
        duration: Duration of firing rate vector in seconds.
        phase: Phase of periodic firing rate vector in radians.
        period: Period of periodic firing rate vector in seconds.
        max_fr: Maximum firing rate in Hz.
        min_fr_before_clipping: Minimum firing rate before clipping in Hz.
        min_fr: Minimum firing rate in Hz. This is used to truncate the firing
            rate sine wave that otherwise would range in
            [min_fr_before_clipping, max_fr].

    Returns:
        firing_rate_vector: Firing rate vector in Hz.
    """
    if phase is None:
        phase = 2 * np.pi * np.random.rand()
    rate_times = np.arange(0, duration)
    freq = 1.0 / period
    sin_wave = 0.5 * (1 + np.sin(phase + freq * rate_times * np.pi * 2))
    firing_rate_vector = (
        min_fr_before_clipping + (max_fr - min_fr_before_clipping) * sin_wave
    )
    firing_rate_vector[firing_rate_vector > max_fr] = max_fr
    firing_rate_vector[firing_rate_vector < min_fr] = min_fr
    return firing_rate_vector


def _get_firing_rate_ramp(
    duration: float,
    max_fr_range: tuple[float, float] = (1, 20),
) -> np.ndarray:
    """Get ramping firing rate vector that disappears at a random time.

    Args:
        duration: Duration of firing rate vector in seconds.
        max_fr_range: Range of maximum firing rate in Hz.

    Returns:
        firing_rate_vector: Firing rate vector in Hz.
    """
    max_fr = np.random.uniform(*max_fr_range)
    disappearance_time = np.random.randint(0, duration)
    firing_rate_vector = np.linspace(max_fr, 0, disappearance_time)
    firing_rate_vector = np.concatenate(
        [firing_rate_vector, np.zeros(duration - disappearance_time)]
    )

    # Flip a coin to make the neuron appear instead of disappear
    if np.random.rand() < 0.5:
        firing_rate_vector = firing_rate_vector[::-1]

    return firing_rate_vector


def _get_firing_rate_stable(
    duration: float, max_fr_range: tuple[float, float] = (1, 20)
) -> np.ndarray:
    """Get stable firing rate vector.

    Args:
        duration: Duration of firing rate vector in seconds.
        max_fr_range: Range of maximum firing rate in Hz.

    Returns:
        firing_rate_vector: Firing rate vector in Hz.
    """
    max_fr = np.random.uniform(*max_fr_range)
    firing_rate_vector = max_fr * np.ones(duration)
    return firing_rate_vector


def _get_spike_train(firing_rate_vector: np.ndarray) -> neo.SpikeTrain:
    """Get spike train from firing rate vector."""
    rate_sig = neo.AnalogSignal(
        firing_rate_vector, units=pq.Hz, sampling_rate=pq.Hz
    )
    spiketrain = elephant.spike_train_generation.NonStationaryPoissonProcess(
        rate_sig, refractory_period=5 * pq.ms
    ).generate_spiketrain()
    spiketrain.annotate(cell_type="E")
    return spiketrain


def _get_spike_train_generator(
    stability: str,
    duration: float,
    unit_num: int,
) -> None | MEArec.SpikeTrainGenerator:
    """Get spike train generator.

    Args:
        stability: Stability of units. Must be 'stable', 'synchronous_periodic',
            'asynchronous_periodic', or 'ramps'.
        duration: Duration of spike train in seconds.
        unit_num: Number of units.

    Returns:
        None | MEArec.SpikeTrainGenerator: Spike train generator.
    """
    print("Getting spike train generator")
    if stability == "stable":
        spgen = None
    elif stability == "synchronous_periodic":
        spiketrains = []
        firing_rate_vector = _get_firing_rate_periodic(
            duration=duration, phase=0
        )
        for _ in range(unit_num):
            spiketrains.append(_get_spike_train(firing_rate_vector))
        spgen = MEArec.SpikeTrainGenerator(spiketrains=spiketrains)
    elif stability == "asynchronous_periodic":
        spiketrains = []
        for _ in range(unit_num):
            firing_rate_vector = _get_firing_rate_periodic(duration=duration)
            spiketrains.append(_get_spike_train(firing_rate_vector))
        spgen = MEArec.SpikeTrainGenerator(spiketrains=spiketrains)
    elif stability == "ramps":
        spiketrains = []
        for _ in range(unit_num // 2):
            firing_rate_vector = _get_firing_rate_ramp(duration=duration)
            spiketrains.append(_get_spike_train(firing_rate_vector))
        for _ in range(unit_num // 2):
            firing_rate_vector = _get_firing_rate_stable(duration=duration)
            spiketrains.append(_get_spike_train(firing_rate_vector))
        spgen = MEArec.SpikeTrainGenerator(spiketrains=spiketrains)
    else:
        raise ValueError(f"Invalid stability: {stability}")

    return spgen


def _plot_drift_vector(drift_dict: dict, recording_dir: Path):
    """Plot drift vector and save figure."""
    fig, ax = plt.subplots(1, 1, figsize=(5, 2))
    ax.plot(
        drift_dict["external_drift_times"],
        drift_dict["external_drift_vector_um"],
    )
    ax.set_xlabel("time (s)", fontsize=12)
    ax.set_ylabel("drift ($\mu$m)", fontsize=12)
    drift_plot_write_path = recording_dir / "drift_plot.png"
    print(f"Saving figure to {drift_plot_write_path}")
    fig.savefig(drift_plot_write_path)
    plt.close(fig)


def generate_data(
    drift_rate: float,
    random_walk_sigma: float,
    random_jump_rate: float,
    unit_num: int,
    homogeneity: str,
    stability: str,
    non_rigidity: float,
    probename: str = "Neuropixels-128",
    duration: float = 1800,
    random_seed: int = 1,
    drift_limits: tuple[float, float] = (-200, 200),
):
    """Generate simulated data.

    This function generates a simulated dataset with the given hyperparameters.
    The dataset is saved in the _SIMULATED_DATASETS_DIR directory.

    Args:
        drift_rate: Drift rate in um/s.
        random_walk_sigma: Standard deviation of random walk in um/sqrt(s).
        random_jump_rate: Rate of Poisson random jumps in 1/s.
        unit_num: Number of units.
        homogeneity: Homogeneity of units. Must be 'uniform' or 'bimodal'.
        stability: Stability of units. Must be 'stable', 'synchronous_periodic',
            'asynchronous_periodic', or 'ramps'.
        non_rigidity: Non-rigidity of units. 0 means rigid (uniform) motion
            across depth. For values != 0, non-rigid motion is a linear function
            of depth.
        probename: Probe name.
        duration: Duration of recording in seconds.
        random_seed: Random seed.
        drift_limits: Drift limits in um.
    """
    np.random.seed(random_seed)

    # Make OrderedDict of hyperparameter values
    hyperparameters = collections.OrderedDict(
        drift_rate=drift_rate,
        random_walk_sigma=random_walk_sigma,
        random_jump_rate=random_jump_rate,
        unit_num=unit_num,
        homogeneity=homogeneity,
        stability=stability,
        non_rigidity=non_rigidity,
    )

    # Check if data already exists
    if _data_exists(hyperparameters):
        print(f"Data already exists for {hyperparameters}")
        return

    # Get drift function
    valid_drift = False
    while not valid_drift:
        drift = drift_lib.Drift(
            duration=duration,
            drift_rate=drift_rate,
            random_walk_sigma=random_walk_sigma,
            random_jump_rate=random_jump_rate,
        )
        valid_drift = np.all(drift.drift_vector < drift_limits[1]) and np.all(
            drift.drift_vector > drift_limits[0]
        )

    # Get templates, generating them if necessary
    tempgen = _get_templates(
        probename=probename,
        drift_limits=drift_limits,
    )

    # Select templates
    print("Selecting template indices")
    templates_ids = _select_template_indices(
        tempgen=tempgen,
        homogeneity=homogeneity,
        unit_num=unit_num,
    )

    # Create drift dictionary
    print("\nCreating drift dictionary")
    external_drift_factors = np.ones(unit_num)
    drift_steps = tempgen.locations.shape[1]
    all_unit_depths = tempgen.locations[:, drift_steps // 2, 2]
    min_unit_depth = np.min(all_unit_depths)
    max_unit_depth = np.max(all_unit_depths)
    unit_depths = np.array([all_unit_depths[i] for i in templates_ids])
    unit_depths_norm = (unit_depths - min_unit_depth) / (
        max_unit_depth - min_unit_depth
    )
    if np.random.rand() < 0.5:
        unit_depths_norm = 1.0 - unit_depths_norm
    if non_rigidity != 0:
        external_drift_factors -= non_rigidity * unit_depths_norm
        print(f"\nexternal_drift_factors: {external_drift_factors}")
        print(f"\nunit_depths: {unit_depths}")
        print(f"\nunit_depths_norm: {unit_depths_norm}")
    drift_dict = {
        "drift_fs": drift.sample_frequency,
        "external_drift_times": drift.drift_times,
        "external_drift_vector_um": drift.drift_vector,
        "external_drift_factors": external_drift_factors,
    }

    # Generate recording
    print("Getting recording parameters")
    recording_params = _get_recording_params(
        duration=duration, unit_num=unit_num, random_seed=random_seed
    )
    print("Generating recording")
    recgen = MEArec.gen_recordings(
        params=recording_params,
        tempgen=tempgen,
        tmp_folder=_TMP_DIR / "MEArec_recordings",
        verbose=True,
        template_ids=templates_ids,
        spgen=_get_spike_train_generator(
            stability=stability,
            duration=duration,
            unit_num=unit_num,
        ),
        drift_dicts=[drift_dict],
    )

    # Get recording path and save recording
    print("Saving recording")
    last_recording_num = max(
        [0] + [int(d.name) for d in _SIMULATED_DATASETS_DIR.iterdir()]
    )
    recording_num = last_recording_num + 1
    recording_name = str(recording_num).zfill(4)
    recording_dir = _SIMULATED_DATASETS_DIR / recording_name
    recording_dir.mkdir(parents=True)
    recording_path = recording_dir / "recording.h5"
    MEArec.save_recording_generator(recgen, filename=recording_path)

    # Save drift function
    drift_dict_write_path = recording_dir / "drift_dict.npy"
    print(f"Saving drift_dict to {drift_dict_write_path}")
    np.save(drift_dict_write_path, drift_dict)

    # Save unit_depths and unit_depths_norm
    print(f"Saving unit_depths and unit_depths_norm")
    np.save(recording_dir / "unit_depths.npy", unit_depths)
    np.save(recording_dir / "unit_depths_norm.npy", unit_depths_norm)

    # Plot drift vector and save figure
    _plot_drift_vector(drift_dict, recording_dir)

    # Run peak detection
    peak_extraction.extract_and_plot_peaks(recording_dir)

    # Save hyperparameters
    hyperparameters_write_path = recording_dir / "hyperparameters.json"
    print(f"Saving hyperparameters to {hyperparameters_write_path}")
    json.dump(hyperparameters, open(hyperparameters_write_path, "w"), indent=4)

    print(f"\n\nFinished {recording_dir}\n\n")


def main(
    drift_rates: tuple[float, ...] = (0.0, 0.1),
    random_walk_sigmas: tuple[float, ...] = (0.0, 1.0, 2.0),
    random_jump_rates: tuple[float, ...] = (0.0, 0.01),
    unit_nums: tuple[float, ...] = (20, 100),
    homogeneities: tuple[str, ...] = ("uniform", "bimodal"),
    stabilities: tuple[str, ...] = (
        "stable",
        "synchronous_periodic",
        "asynchronous_periodic",
        "ramps",
    ),
    non_rigidities: tuple[float, ...] = (0.0, 0.5),
):
    """Generate simulated data for each parameter combination.

    Datasets that have already been generated will not be re-generated. All
    datasets will be saved in the _SIMULATED_DATASETS_DIR directory.

    Each argument should be a comma-separated list or tuple of values. A
    dataset will be generated for each possible combinations of the values.

    Args:
        drift_rates: Drift rates in um/s.
        random_walk_sigmas: Standard deviations of random walks in um/sqrt(s).
        random_jump_rates: Rates of Poisson random jumps in 1/s.
        unit_nums: Number of units.
        homogeneities: Homogeneity of units.
        stabilities: Stability of units.
        non_rigidities: Non-rigidity of units. 0 means rigid (uniform) motion
            across depth. For values != 0, non-rigid motion is a linear function
            of depth.
    """
    # Convert comma-separated strings to lists, which can happen for command
    # line inputs
    if isinstance(drift_rates, str):
        drift_rates = [float(x) for x in drift_rates.split(",")]
    if isinstance(random_walk_sigmas, str):
        random_walk_sigmas = [float(x) for x in random_walk_sigmas.split(",")]
    if isinstance(random_jump_rates, str):
        random_jump_rates = [float(x) for x in random_jump_rates.split(",")]
    if isinstance(unit_nums, str):
        unit_nums = [int(x) for x in unit_nums.split(",")]
    if isinstance(homogeneities, str):
        homogeneities = homogeneities.split(",")
    if isinstance(stabilities, str):
        stabilities = stabilities.split(",")
    if isinstance(non_rigidities, str):
        non_rigidities = [float(x) for x in non_rigidities.split(",")]

    # Generate data for each parameter combination
    sweeps = collections.OrderedDict(
        drift_rate=drift_rates,
        random_walk_sigma=random_walk_sigmas,
        random_jump_rate=random_jump_rates,
        unit_num=unit_nums,
        homogeneity=homogeneities,
        stability=stabilities,
        non_rigidity=non_rigidities,
    )
    parameter_names = sweeps.keys()
    for parameter_values in itertools.product(*sweeps.values()):
        generate_data(**dict(zip(parameter_names, parameter_values)))

    print("\n\nDONE\n\n")


if __name__ == "__main__":
    main(**dict(arg.split("=") for arg in sys.argv[1:]))
