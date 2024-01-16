from functools import partial
from typing import Optional
import jax
import jax.numpy as jnp
import flax.linen as nn


def init_A_S4D(key, mode: str, channels: int, state_dim: int):
    """intialization for diagonal SSM https://arxiv.org/abs/2206.11893"""
    if mode == "real":
        A = -1.0 * (1 + jnp.arange(state_dim)) + 0j
    elif mode == "lin":
        A = -0.5 + 1j * jnp.arange(state_dim // 2)
    else:
        raise ValueError(f"Unknown mode {mode}")
    A = jnp.broadcast_to(A, (channels, *A.shape))
    return jnp.log(-A.real), A.imag


class S6(nn.Module):
    state_dim: int
    complex: bool
    chunk_size: int = 2048 * 64 * 64

    def get_ssm_params(self, x: jax.Array):
        lenght, channels = x.shape
        state_dim = self.state_dim // 2 if self.complex else self.state_dim

        if self.complex:
            lognegAreal, Aimag = self.param("A", init_A_S4D, "lin", channels, self.state_dim)
            A = -jnp.exp(lognegAreal) + 1j * Aimag
        else:
            lognegAreal, A_imag = self.param("A", self.init_A_S4D, "real", channels)
            A = -jnp.exp(lognegAreal)
        B = 1 + nn.Dense(state_dim, param_dtype=A.dtype)(x)
        C = nn.Dense(state_dim, param_dtype=A.dtype)(x)
        dt = nn.softplus(jnp.log(jnp.exp(0.01) - 1) + nn.Dense(channels)(nn.Dense(1)(x)))

        # ZOH discretization and prepare scan inputs
        At = jnp.exp(jnp.einsum("cs, lc->lcs", A, dt))
        ut = jnp.einsum("lcs, cs, ls, lc->lcs", (At - 1), 1 / A, B, x)
        Ct = C.reshape((lenght, 1, state_dim))

        return At, ut, Ct

    @nn.compact
    def __call__(self, x: jax.Array):
        lenght, channels = x.shape
        At, ut, Ct = self.get_ssm_params(x)

        def aggregate_ssm(step1, step2):
            """fuse two ssm steps into one"""
            (A1, u1), (A2, u2) = step1, step2
            return (A2 * A1, A2 * u1 + u2)

        def scan_chunked_ssm(ht, ssm_chunk):
            At, ut, Ct = ssm_chunk
            At, ut = jax.lax.associative_scan(aggregate_ssm, (At, ut))
            ht = At * ht + ut
            yt = jnp.einsum("lcs, lcs->lc", Ct, ht)
            return ht[-1], yt

        # chunk ssm along time axis to limit memory usage
        max_length = min((lenght, self.chunk_size // (channels * self.state_dim)))
        At = At.reshape((-1, max_length, *At.shape[1:]))
        ut = ut.reshape((-1, max_length, *ut.shape[1:]))
        Ct = Ct.reshape((-1, max_length, *Ct.shape[1:]))

        h0 = jnp.zeros_like(At[0, 0, :])
        ht, y = jax.lax.scan(scan_chunked_ssm, h0, (At, ut, Ct))
        y = y.reshape((lenght, channels))

        return (y + jnp.conjugate(y)).real


class S4(S6):
    def get_ssm_params(self, x: jax.Array):
        lenght, channels = x.shape
        state_dim = self.state_dim // 2 if self.complex else self.state_dim

        if self.complex:
            lognegAreal, Aimag = self.param("A", init_A_S4D, "lin", channels, self.state_dim)
            A = -jnp.exp(lognegAreal) + 1j * Aimag
        else:
            lognegAreal, A_imag = self.param("A", self.init_A_S4D, "real", channels)
            A = -jnp.exp(lognegAreal)

        B = self.param("B", nn.initializers.ones, A.shape, A.dtype)
        C = self.param("C", nn.initializers.ones, A.shape, A.dtype)
        dt0 = jnp.log(jnp.exp(1 / self.sample_rate) - 1)  # inverse of softplus
        dt = nn.softplus(self.param("dt", nn.initializers.constant(dt0), (channels,)))

        # ZOH discretization and prepare scan inputs
        At = jnp.exp(jnp.einsum("cs, c->cs", A, dt))
        ut = jnp.einsum("cs, cs, cs, lc->lcs", (At - 1), 1 / A, B, x)
        At = jnp.broadcast_to(At, (lenght, channels, state_dim))
        Ct = jnp.broadcast_to(C, (lenght, channels, state_dim))

        return At, ut, Ct


class S4Block(nn.Module):
    state_dim: int
    complex: bool

    @nn.compact
    def __call__(self, x: jax.Array):
        lenght, channels = x.shape

        # first stage: ssm + nonlinearity + dense
        h = nn.RMSNorm()(x)
        h = S4(self.state_dim, self.complex)(h)
        h = nn.gelu(h)
        h = nn.Dense(channels)(h)
        x = x + h

        # second stage: MLP
        h = nn.RMSNorm()(x)
        h = nn.Dense(4 * channels)(h)
        h = nn.gelu(h)
        h = nn.Dense(channels)(h)
        return x


class MambaBlock(nn.Module):
    state_dim: int
    complex: bool

    @nn.compact
    def __call__(self, x: jax.Array):
        lenght, channels = x.shape
        h = nn.RMSNorm()(x)

        # gate path
        g = nn.Dense(2 * channels, use_bias=False)(h)
        g = nn.gelu(g)

        # residual path
        h = nn.Dense(2 * channels, use_bias=False)(h)
        h = nn.Conv(2 * channels, kernel_size=(self.state_dim,), padding="CAUSAL")(h)
        h = nn.gelu(h)
        h = S6(self.state_dim, self.complex)(h)
        h = nn.Dense(channels, use_bias=False)(h * g)

        return x + h
