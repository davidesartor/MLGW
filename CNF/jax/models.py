import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
import flax.linen as nn


class FCBlock(nn.Module):
    expand_factor: int = 4
    depth: int = 1

    @nn.compact
    def __call__(self, x):
        *B, C = x.shape
        for _ in range(self.depth):
            x = nn.Dense(C * self.expand_factor)(x)
            x = nn.silu(x)
        x = nn.Dense(C)(x)
        return x


class ZeroInitDense(nn.Dense):
    kernel_init: nn.initializers.Initializer = nn.initializers.zeros
    bias_init: nn.initializers.Initializer = nn.initializers.zeros


class AdaLayerNorm(nn.Module):
    @nn.compact
    def __call__(self, x, scale, shift):
        x = nn.LayerNorm(use_scale=False, use_bias=False)(x)
        return x * (1 + scale) + shift


class DiTBlock(nn.Module):
    heads: int

    def modulation(self, x):
        x = nn.Dense(x.shape[-1])(x)
        x = nn.silu(x)
        x = ZeroInitDense(x.shape[-1] * 6)(x)
        return jnp.split(x[..., None, :], 6, axis=-1)

    @nn.compact
    def __call__(self, x, c):
        scale1, shift1, gate1, scale2, shift2, gate2 = self.modulation(c)
        attention = nn.MultiHeadAttention(self.heads)
        x = x + gate1 * attention(AdaLayerNorm()(x, scale1, shift1))
        x = x + gate2 * FCBlock()(AdaLayerNorm()(x, scale2, shift2))
        return x


class DownSample(nn.Module):
    factor: int

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(x.shape[-1], self.factor, self.factor)(x)
        return x


class Patchify(nn.Module):
    dim: int
    patch_size: int = 16

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(self.dim, self.patch_size, self.patch_size)(x)
        x = nn.silu(x)
        x = nn.Conv(self.dim, 1)(x)
        return x


class PosEmbed(nn.Module):
    embed_dim: int
    max_period: int = 10000

    def positional_embedding(self, k):
        log_freqs = -np.log(self.max_period) * np.arange(0, 1, 2 / self.embed_dim)
        x = k[:, None] * np.exp(log_freqs)[None, :]
        x = jnp.concatenate([jnp.sin(x), jnp.cos(x)], axis=-1)
        return x

    @nn.compact
    def __call__(self, x):
        *B, L, C = x.shape
        x = self.positional_embedding(jnp.arange(L))
        if len(B) > 0:
            x = jnp.expand_dims(x, list(range(len(B))))
        return x


class TimeEmbed(PosEmbed):
    embed_dim: int
    max_period: int = 100

    @nn.compact
    def __call__(self, t):
        x = self.positional_embedding(t * self.max_period)
        x = FCBlock()(x)
        return x


class ParamEmbed(nn.Module):
    embed_dim: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.embed_dim)(x)
        x = nn.silu(x)
        x = FCBlock()(x)
        return x
