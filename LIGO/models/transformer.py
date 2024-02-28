import jax
import jax.numpy as jnp
import flax.linen as nn


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
        heads = nn.vmap(
            AttentionHead,
            variable_axes={"params": -1},
            split_rngs={"params": True},
            in_axes=None,  # type: ignore
            out_axes=-1,
            axis_size=self.n_heads,
        )
        x = heads(self.channels // self.n_heads)(x)
        x = jnp.reshape(x, (x.shape[0], -1))
        return x


class TransformerBlock(nn.Module):
    head_size: int
    n_heads: int

    @nn.compact
    def __call__(self, x: jax.Array):
        lenght, channels = x.shape

        h = nn.RMSNorm()(x)
        h = MultiAttentionHead(self.head_size, self.n_heads)(h)
        h = nn.Dense(channels)(h)
        x = x + h

        h = nn.RMSNorm()(x)
        h = nn.Dense(4 * channels)(h)
        h = nn.gelu(h)
        h = nn.Dense(channels)(h)
        return x + h
