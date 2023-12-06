from dataclasses import dataclass
import numpy as np
import scipy
import chirping_binary
import noise


@dataclass
class DatasetChirping:
    episode_duration_s: float
    sample_rate_Hz: float
    n_sources: int = 1
    episodes_per_epoch: int = 1024
    stft_window_size: int = 32

    @property
    def sample_period_s(self):
        return 1 / self.sample_rate_Hz

    @property
    def times(self):
        return np.arange(0, self.episode_duration_s, self.sample_period_s)

    def __len__(self):
        return self.episodes_per_epoch

    def __getitem__(self, index):
        parameters = chirping_binary.Parameters.sample(time_range=(0, self.episode_duration_s))
        source_signal = chirping_binary.h_plus(times=self.times, parameters=parameters)
        noise_signal = noise.generate(self.sample_period_s, len(self.times), noise.LIGOL())
        f, t, psd = scipy.signal.stft(
            source_signal + noise_signal,
            fs=self.sample_rate_Hz,
            nperseg=self.stft_window_size,
            noverlap=self.stft_window_size // 4,
        )
        return (np.abs(psd), np.array(parameters))
