import jax
import jax.numpy as jnp
import flax.linen as nn


class MLP(nn.Module):
    hidden_dim: int
    output_dim: int

    @nn.compact
    def __call__(self, x: jax.Array):
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.gelu(x)
        x = nn.Dense(self.output_dim)(x)
        return x
