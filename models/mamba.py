import jax
import jax.numpy as jnp
import flax.linen as nn


class SSM(nn.Module):
    state_dim: int
    dt: float | None = None

    @nn.compact
    def __call__(self, x: jax.Array):
        # initialize A (hungry hungry hippos style)
        Ainit = 1 + jnp.arange(self.state_dim, dtype=jnp.float32)
        A = -jnp.exp(self.param("logA", lambda k: jnp.log(Ainit)))
        B = nn.Dense(self.state_dim, use_bias=False)(x)
        C = nn.Dense(self.state_dim, use_bias=False)(x)

        # discretization
        dt = self.dt or nn.softplus(nn.Dense(1, use_bias=True)(x))
        A, B = jnp.exp(A * dt), (jnp.exp(A * dt) - 1) * B / A

        def scan_fn(el1, el2):
            (A1, B1), (A2, B2) = el1, el2
            return A1 * A2, A1 * B2

        _, Ak_B = jax.lax.associative_scan(scan_fn, (A, B))
        x = jnp.einsum("ls,ls,lc->lc", C, Ak_B, x)
        return x


class MambaBlock(nn.Module):
    state_dim: int

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
        x = SSM(self.state_dim)(x)
        x = nn.silu(x)
        x = SSM(self.state_dim)(x)
        return x
