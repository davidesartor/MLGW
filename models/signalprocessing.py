import jax
import jax.numpy as jnp
import flax.linen as nn
from .mamba import MambaBlock, S4DBlock, S6DReal, S6DComplex, S4DReal, S4DComplex


class S4DSP(nn.Module):
    sample_rate_Hz: float
    state_dim: int
    n_layers: int = 1
    ssm_module: type[nn.Module] = S4DComplex

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        for _ in range(self.n_layers):
            x = S4DBlock(self.state_dim, self.sample_rate_Hz, self.ssm_module)(x)
        return x


class MambaSP(nn.Module):
    sample_rate_Hz: float
    state_dim: int
    n_layers: int = 1
    ssm_module: type[nn.Module] = S6DReal

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        for _ in range(self.n_layers):
            x = MambaBlock(self.state_dim, self.sample_rate_Hz, self.ssm_module)(x)
        return x
