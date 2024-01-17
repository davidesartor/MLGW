from functools import partial
from typing import Any
import jax
import jax.numpy as jnp
import flax.linen as nn
from .mamba import S4, S6, MambaBlock, S4Block


class DownPool(nn.Module):
    pool_factor: int = 4
    expand_factor: int = 2

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        lenght, channels = x.shape
        x = x.reshape((lenght // self.pool_factor, -1))
        x = nn.Dense(self.expand_factor * channels)(x)
        return x


class UpPool(nn.Module):
    pool_factor: int = 4
    expand_factor: int = 2

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        lenght, channels = x.shape
        x = nn.Dense(channels * self.pool_factor // self.expand_factor)(x)
        x = x.reshape((lenght * self.pool_factor, -1))
        return x


class SaShiMi(nn.Module):
    stages: int
    stage_layers: int
    hidden_channels: int
    hidden_state_dim: int
    complex = True

    def transform(self, x: jax.Array) -> jax.Array:
        for _ in range(self.stage_layers):
            x = S4Block(self.hidden_state_dim, self.complex)(x)
        return x

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        def forward(stages: int, x: jax.Array) -> jax.Array:
            if stages == 1:
                return x + self.transform(x)

            x = self.transform(x)

            # residual path: downsample + recursive call + upsample
            h = DownPool()(x)
            h = forward(stages - 1, h)
            h = jnp.roll(h.at[-1, :].set(0.0), 1, axis=0)  # recover causality
            h = UpPool()(h)
            x = x + h

            x = self.transform(x)
            return x

        lenght, channels = x.shape
        x = nn.Dense(self.hidden_channels, use_bias=False)(x)
        x = forward(self.stages, x)
        x = nn.Dense(channels)(x)
        return x


class Mamba(SaShiMi):
    def transform(self, x: jax.Array) -> jax.Array:
        for _ in range(self.stage_layers):
            x = MambaBlock(self.hidden_state_dim, self.complex)(x)
        return x
