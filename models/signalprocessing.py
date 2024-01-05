import jax
import jax.numpy as jnp
import flax.linen as nn
from .mamba import MambaBlock


class MambaSP(nn.Module):
    sample_rate_Hz: float
    state_dim: int
    n_layers: int = 1

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        x = x[..., None]
        for _ in range(self.n_layers):
            x = MambaBlock(self.state_dim, self.sample_rate_Hz)(x)
        x = x[..., 0]
        return x
