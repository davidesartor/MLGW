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


class Unet(nn.Module):
    stages: int

    def transform(self, x: jax.Array) -> jax.Array:
        return x

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        pooled = [x]
        for _ in range(self.stages - 1):
            pooled.append(DownPool()(pooled[-1]))

        h = pooled[-1]
        h = self.transform(h)
        for p in reversed(pooled[:-1]):
            h = jnp.roll(h.at[-1, :].set(0.0), 1, axis=0)  # recover causality
            h = UpPool()(h)
            h = h + p
            h = self.transform(h)
        return h


class SaShiMi(Unet):
    stage_layers: int
    hidden_state_dim: int
    complex = True

    def transform(self, x: jax.Array) -> jax.Array:
        for _ in range(self.stage_layers):
            x = S4Block(self.hidden_state_dim, self.complex)(x)
        return x


class Mamba(Unet):
    stage_layers: int
    hidden_state_dim: int
    complex = True

    def transform(self, x: jax.Array) -> jax.Array:
        for _ in range(self.stage_layers):
            x = MambaBlock(self.hidden_state_dim, self.complex)(x)
        return x
