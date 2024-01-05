import jax
import jax.numpy as jnp
import flax.linen as nn
from .generic import MLP


class SSMS4(nn.Module):
    state_dim: int
    sample_rate: float

    @nn.compact
    def __call__(self, x: jax.Array):
        lenght, channels = x.shape

        # initialize A (hungry hungry hippos style)
        Ainit = 1 + jnp.arange(self.state_dim, dtype=jnp.float32)
        A = -jnp.exp(self.param("logA", lambda k: jnp.log(Ainit)))
        B = self.param("B", nn.initializers.ones, (self.state_dim,))
        C = self.param("C", nn.initializers.normal(), (self.state_dim,))

        # discretization
        dt = 1 / self.sample_rate
        A, B = jnp.exp(A * dt), (jnp.exp(A * dt) - 1) * B / A

        def ssm_scan(h, x):
            h = jnp.einsum("s, cs->cs", A, h) + jnp.einsum("s, c->cs", B, x)
            y = jnp.einsum("s, cs->c", C, h)
            return h, y

        h, x = jax.lax.scan(ssm_scan, jnp.zeros((channels, self.state_dim)), x)
        return x


class SSMS6(nn.Module):
    state_dim: int
    sample_rate: float

    @nn.compact
    def __call__(self, x: jax.Array):
        lenght, channels = x.shape

        # initialize A (hungry hungry hippos style)
        Ainit = 1 + jnp.arange(self.state_dim, dtype=jnp.float32)
        A = -jnp.exp(self.param("logA", lambda k: jnp.log(Ainit)))
        B = 1 + nn.Dense(self.state_dim, use_bias=False)(x)
        C = nn.Dense(self.state_dim, use_bias=False)(x)

        # discretization
        dt = nn.softplus(nn.Dense(1, bias_init=nn.initializers.constant(1 / self.sample_rate))(x))
        A, B = jnp.exp(A * dt), (jnp.exp(A * dt) - 1) * B / A

        def ssm_scan(h, state):
            A, B, C, x = state
            h = jnp.einsum("s, cs->cs", A, h) + jnp.einsum("s, c->cs", B, x)
            y = jnp.einsum("s, cs->c", C, h)
            return h, y

        h, x = jax.lax.scan(ssm_scan, jnp.zeros((channels, self.state_dim)), (A, B, C, x))
        return x


class MambaBlock(nn.Module):
    state_dim: int
    sample_rate: float

    @nn.compact
    def __call__(self, x: jax.Array):
        lenght, channels = x.shape
        residual = x
        x = nn.RMSNorm()(x)
        x = self.ssm(x) * self.gate(x)
        x = nn.Dense(channels, use_bias=False)(x)
        x = x + residual
        return x

    def gate(self, x: jax.Array):
        lenght, channels = x.shape
        x = nn.Dense(2 * channels, use_bias=False)(x)
        x = nn.silu(x)
        return x

    def ssm(self, x: jax.Array):
        lenght, channels = x.shape
        x = nn.Dense(2 * channels, use_bias=False)(x)
        x = nn.Conv(2 * channels, kernel_size=(self.state_dim,), padding="CAUSAL")(x)
        x = nn.silu(x)
        x = SSMS4(self.state_dim, self.sample_rate)(x)
        return x
