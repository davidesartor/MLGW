import torch
import numpy as np
import random
from dataclasses import dataclass


@dataclass
class SincSource:
    theta: np.ndarray
    lambd: float

    def signal(self, t: np.ndarray) -> np.ndarray:
        period, shift = self.theta[0], self.theta[1]
        return np.sinc(2 * (t - shift) / period) * self.lambd

    @classmethod
    def sample(cls, n_sources=1) -> list["SincSource"]:
        if n_sources == 1:
            lambd = random.uniform(0.5, 2.0)
            period = random.uniform(1.0, 2.0)
            shift = random.uniform(-5, 5)
            return [cls(theta=np.array([period, shift]), lambd=lambd)]
        if n_sources > 1:
            return [cls.sample()[0] for _ in range(n_sources)]
        else:
            raise ValueError("n_sources must be positive")


def create_batch(batch_size: int, n_sources: int, t: np.ndarray):
    sources = [SincSource.sample(n_sources) for _ in range(batch_size)]
    signals = [np.sum([source.signal(t) for source in ss], axis=0) for ss in sources]
    signals = torch.as_tensor(np.array(signals), dtype=torch.float32).unsqueeze(1)
    return signals, sources


def unpack_params(params: torch.Tensor):
    # params shape: (n_batch, n_sources, n_source_params + 1)
    log_periods = params[:, :, 0]
    shifts = params[:, :, 1]
    log_lambdas = params[:, :, 2]
    return log_periods.exp(), shifts, log_lambdas.exp()


def get_sources_from(params: torch.Tensor) -> list[list[SincSource]]:
    periods, shifts, lambdas = unpack_params(params)

    periods = periods.detach().cpu().numpy()
    shifts = shifts.detach().cpu().numpy()
    lambdas = lambdas.detach().cpu().numpy()
    thetas = np.stack([periods, shifts], axis=-1)

    return [
        [
            SincSource(theta=theta, lambd=lambd)
            for theta, lambd in zip(thetas_batch, lambdas_batch)
        ]
        for thetas_batch, lambdas_batch in zip(thetas, lambdas)
    ]


def decode(params: torch.Tensor, times: torch.Tensor):
    periods, shifts, lambdas = unpack_params(params)

    # needed to make broadcasting work (avoids loops)
    periods = periods.unsqueeze(0)  # shape: (1, n_batch, n_sources)
    shifts = shifts.unsqueeze(0)  # shape: (1, n_batch, n_sources)
    lambdas = lambdas.unsqueeze(0)  # shape: (1, n_batch, n_sources)
    times = times.unsqueeze(1).unsqueeze(1)  # shape: (n_times, 1, 1)

    # shape (n_times, n_batch, n_sources)
    signals = torch.sinc(2 * (times - shifts) / periods) * lambdas
    return signals.sum(dim=-1).T  # shape: (n_batch, n_times)
