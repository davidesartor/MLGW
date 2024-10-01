from dataclasses import dataclass
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np


@dataclass
class SinusoidDataset(Dataset):
    dataset_size: int = 1024 * 1024

    n_sources: int = 1
    A_range: tuple[float, float] = (1e0, 1e1)
    omega_range: tuple[float, float] = (1e-2, 1e-1)
    phi_range: tuple[float, float] = (0.0, 2 * np.pi)
    dual_channel: bool = False

    observation_time: float = 1024.0
    sampling_rate: float = 1.0
    white_noise_std: float = 1e1

    def __len__(self):
        return self.dataset_size

    def dataloader(self, *args, **kwargs):
        return DataLoader(self, *args, **kwargs)

    def sample_diffusion_time(self):
        logit = np.random.randn(1)
        return 1 / (1 + np.exp(-logit))

    def sample_params(self):
        n = self.n_sources
        A_min, A_max = self.A_range
        omega_min, omega_max = self.omega_range
        phi_min, phi_max = self.phi_range
        log_A = np.random.uniform(np.log(A_min), np.log(A_max), n)
        log_omega = np.random.uniform(np.log(omega_min), np.log(omega_max), n)
        phi = np.random.uniform(phi_min, phi_max, n)
        return np.array([np.exp(log_A), np.exp(log_omega), phi])

    def clean_signal(self, params):
        A, omega, phi = params
        t = np.arange(0.0, self.observation_time, self.sampling_rate)[:, None]
        h1 = (A * np.sin(omega * t + phi)).sum(axis=-1, keepdims=True)
        if not self.dual_channel:
            return h1
        h2 = (A * np.cos(omega * t + phi)).sum(axis=-1, keepdims=True)
        return np.concatenate([h1, h2], axis=-1)

    def datastream(self, clean_signal):
        noise = np.random.normal(0, self.white_noise_std, clean_signal.shape)
        return clean_signal + noise

    def __getitem__(self, idx=None):
        t = self.sample_diffusion_time()
        params = self.sample_params()
        clean = self.clean_signal(params)
        noisy = self.datastream(clean)

        to_tensor = lambda x: torch.as_tensor(x, dtype=torch.float32)
        return tuple(map(to_tensor, (t, params, clean, noisy)))


def double_batch(dataset_cls):
    class DoubleBatchDataset(dataset_cls):
        def __getitem__(self, idx=None):
            batch1 = super().__getitem__(idx)
            batch2 = super().__getitem__(idx)
            return (*batch1, *batch2)

    return DoubleBatchDataset
