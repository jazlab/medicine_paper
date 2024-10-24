"""Drift generator.

This file contains the Drift class, which generates a drift vector over time.
"""

import numpy as np


class Drift:
    """Drift class."""

    def __init__(
        self,
        duration: float = 1800,
        sample_frequency: float = 10,
        drift_rate: float = 0,
        random_walk_sigma: float = 0,
        random_jump_rate: float = 0,
        max_jump_magnitude: float = 50,
        random_seed: float = 1,
    ):
        """Constructor.

        Args:
            duration: Duration of drift in seconds.
            sample_frequency: Sample frequency in Hz.
            drift_rate: Drift rate in um/s.
            random_walk_sigma: Standard deviation of random walk in um/sqrt(s).
            random_jump_rate: Rate of Poisson random jumps in 1/s.
            max_jump_magnitude: Maximum jump magnitude in um. Random jumps are
                uniformly distributed between -max_jump_magnitude and
                max_jump_magnitude.
            random_seed: Random seed.
        """
        self._duration = duration
        self._sample_frequency = sample_frequency
        self._drift_rate = drift_rate
        self._random_walk_sigma = random_walk_sigma
        self._random_jump_rate = random_jump_rate
        self._max_jump_magnitude = max_jump_magnitude
        np.random.seed(random_seed)

        # Create drift times
        self._drift_times = np.arange(0, duration, 1 / sample_frequency)

        # Create drift vector
        drift_vector = (
            self._constant_drift() + self._random_walk() + self._random_jumps()
        )
        drift_vector -= 0.5 * (np.max(drift_vector) + np.min(drift_vector))
        self._drift_vector = drift_vector

    def _constant_drift(self):
        return self._drift_rate * self._drift_times

    def _random_walk(self):
        samples = (1 / np.sqrt(self.sample_frequency)) * np.random.normal(
            0.0, self._random_walk_sigma, size=len(self._drift_times)
        )
        random_walk = np.zeros_like(self._drift_times)
        for i in range(len(self._drift_times) - 1):
            random_walk[i + 1] = random_walk[i] + samples[i]
        return random_walk

    def _random_jumps(self):
        deltas = np.zeros_like(self._drift_times)
        prob_jump_per_sample = self._random_jump_rate / self._sample_frequency
        for i in range(len(self._drift_times)):
            if np.random.uniform(0.0, 1.0) < prob_jump_per_sample:
                jump = self._max_jump_magnitude * np.random.uniform(-1, 1)
                deltas[i:] += jump
        return deltas

    @property
    def duration(self):
        return self._duration

    @property
    def sample_frequency(self):
        return self._sample_frequency

    @property
    def drift_times(self):
        return self._drift_times

    @property
    def drift_vector(self):
        return self._drift_vector
