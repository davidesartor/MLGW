import jax
import jax.numpy as jnp
import flax.linen as nn
from .generic import MLP


class AttentionHead(nn.Module):
    channels: int

    @nn.compact
    def __call__(self, x: jax.Array):
        keys = nn.Dense(self.channels, use_bias=False)(x)
        values = nn.Dense(self.channels, use_bias=False)(x)
        queries = nn.Dense(self.channels, use_bias=False)(x)

        causal_mask = jnp.tril(jnp.ones((len(x), len(x))))
        weights: jax.Array = keys @ queries.T / jnp.sqrt(self.channels)
        weights = jax.nn.softmax(jnp.where(causal_mask == 1, weights, -jnp.inf), axis=-1)
        return weights @ values


class MultiAttentionHead(nn.Module):
    channels: int
    n_heads: int = 1

    @nn.compact
    def __call__(self, x: jax.Array):
        return jnp.concatenate(
            [AttentionHead(self.channels // self.n_heads)(x) for _ in range(self.n_heads)], axis=-1
        )


class TransformerBlock(nn.Module):
    head_size: int
    n_heads: int

    @nn.compact
    def __call__(self, x: jax.Array):
        x = x + self.attention(x)
        x = x + self.feedforward(x)
        return x

    def attention(self, x: jax.Array):
        time, channels = x.shape
        x = nn.RMSNorm()(x)
        x = MultiAttentionHead(self.head_size, self.n_heads)(x)
        x = nn.Dense(channels)(x)
        return x

    def feedforward(self, x: jax.Array):
        time, channels = x.shape
        x = nn.RMSNorm()(x)
        x = MLP(4 * channels, channels)(x)
        return x
