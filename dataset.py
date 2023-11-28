from dataclasses import dataclass
import numpy as np
import chirping_binary
import noise


@dataclass
class DatasetChirping:
    n_measurements: int
    sampling_period: float
    n_sources: int = 1
    examples_per_epoch: int = 1024

    @property
    def times(self):
        return np.arange(0, self.n_measurements * self.sampling_period, self.sampling_period)

    @property
    def time_range(self):
        return (0, self.times[-1])

    def __getitem__(self, index):
        parameters = chirping_binary.Parameters.sample(time_range=self.time_range)
        source_signal = chirping_binary.h_plus(times=self.times, parameters=parameters)
        noise_signal = noise.generate(self.sampling_period, self.n_measurements, noise.LIGOL())
        return (source_signal + noise_signal, parameters)

    def __len__(self):
        return self.examples_per_epoch
