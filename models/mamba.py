import jax
import jax.numpy as jnp
import flax.linen as nn
from .generic import MLP


class SSM(nn.Module):
    state_dim: int
    sample_rate: float
    selective: bool = True

    @nn.compact
    def __call__(self, x: jax.Array):
        lenght, channels = x.shape

        # initialize A (hungry hungry hippos style)
        Ainit = 1 + jnp.arange(self.state_dim, dtype=jnp.float32)
        A = -jnp.exp(self.param("logA", lambda k: jnp.log(Ainit)))

        if self.selective:
            dt = nn.softplus(1 / self.sample_rate + nn.Dense(channels)(x))
            B = 1 + nn.Dense(self.state_dim)(x)
            C = nn.Dense(self.state_dim)(x)
        else:
            dt = 1 / self.sample_rate * jnp.ones((lenght, channels))
            B = self.param("B", nn.initializers.ones, (1, self.state_dim))
            C = self.param("C", nn.initializers.normal(), (1, self.state_dim))

        # discretization
        Adiscr = jnp.exp(jnp.einsum("s, lc->lcs", A, dt))
        Bdiscr = jnp.exp(jnp.einsum("lcs, ls->lcs", Adiscr - 1, B / A))

        def ssm_scan(p1, p2):
            (A1, c1), (A2, c2) = p1, p2
            return (A2 * A1, c2 + A2 * c1)

        u = jnp.einsum("lcs, lc->lcs", Bdiscr, x)
        _, h = jax.lax.associative_scan(ssm_scan, (Adiscr, u))
        y = jnp.einsum("ls, lcs->lc", C, h)
        return y


class S4Block(nn.Module):
    state_dim: int
    sample_rate: float

    @nn.compact
    def __call__(self, x: jax.Array):
        lenght, channels = x.shape
        residual = x
        x = nn.RMSNorm()(x)
        x = SSM(self.state_dim, self.sample_rate, selective=False)(x)
        x = nn.Dense(channels)(x)
        x = x + residual
        return x


class MambaBlock(nn.Module):
    state_dim: int
    sample_rate: float = 0.1
    selective: bool = True

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
        x = SSM(self.state_dim, self.sample_rate, self.selective)(x)
        return x
