from typing import Optional
import jax
import jax.numpy as jnp
import flax.linen as nn
import einops


class S6(nn.Module):
    state_dim: int
    complex: bool

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        At, ut, Ct = self.get_discretized_ssm(x)

        def aggregate_ssm(step1, step2):
            """fuse two ssm steps into one"""
            (A1, u1), (A2, u2) = step1, step2
            return (A2 * A1, A2 * u1 + u2)

        At, ht = jax.lax.associative_scan(aggregate_ssm, (At, ut))
        y = einops.einsum(Ct, ht, "l c s, l c s -> l c")
        if self.complex:
            y = (y + jnp.conjugate(y)).real
        return y

    def init_A_S4D(self, key, channels: int):
        """intialization for diagonal SSM https://arxiv.org/abs/2206.11893"""
        if self.complex:
            A = -0.5 + 1j * jnp.arange(self.state_dim // 2)
        else:
            A = -1.0 * (1 + jnp.arange(self.state_dim)) + 0j
        A = einops.repeat(A, "state -> channel state", channel=channels)
        return jnp.log(-A.real), A.imag

    def get_discretized_ssm(self, x: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
        lenght, channels = x.shape
        lognegAreal, Aimag = self.param("A", self.init_A_S4D, channels)
        A = -jnp.exp(lognegAreal) + (1j * Aimag if self.complex else 0.0)
        B = nn.Dense(A.shape[-1], param_dtype=A.dtype)(x)
        C = nn.Dense(A.shape[-1], param_dtype=A.dtype)(x)
        dt = nn.softplus(jnp.log(jnp.exp(0.01) - 1) + nn.Dense(channels)(nn.Dense(1)(x)))

        # ZOH discretization
        At = jnp.exp(einops.einsum(A, dt, "c s, l c -> l c s"))
        Bt = einops.einsum(At - 1, B, 1 / A, "l c s, l s, c s -> l c s")
        Ct = einops.rearrange(C, "l s -> l 1 s")
        ut = einops.einsum(Bt, x, "l c s, l c -> l c s")
        return At, ut, Ct


class S4(S6):
    def get_discretized_ssm(self, x: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
        lenght, channels = x.shape
        lognegAreal, Aimag = self.param("A", self.init_A_S4D, channels)
        A = -jnp.exp(lognegAreal) + (1j * Aimag if self.complex else 0.0)
        B = self.param("B", nn.initializers.ones, A.shape, A.dtype)
        C = self.param("C", nn.initializers.uniform(1), A.shape, A.dtype)
        dt0 = jnp.log(jnp.exp(0.01) - 1)  # inverse of softplus
        dt = nn.softplus(self.param("dt", nn.initializers.constant(dt0), (channels,)))

        # ZOH discretization
        At = jnp.exp(einops.einsum(A, dt, "c s, c -> c s"))
        Bt = einops.einsum(At - 1, B, 1 / A, "c s, c s, c s -> c s")
        ut = einops.einsum(Bt, x, "c s, l c -> l c s")
        At = einops.repeat(At, "c s -> l c s", l=lenght)
        Ct = einops.repeat(C, "c s -> l c s", l=lenght)
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
        # independent convolutions for each channel
        Conv = nn.vmap(nn.Conv, variable_axes={"params": 0}, split_rngs={"params": True})

        lenght, channels = x.shape
        h = nn.RMSNorm()(x)

        # gate path
        g = nn.Dense(2 * channels, use_bias=False)(h)
        g = nn.gelu(g)

        # residual path
        h = nn.Dense(2 * channels, use_bias=False)(h)
        h = einops.rearrange(h, "l c -> c l 1")
        h = Conv(1, kernel_size=(self.state_dim,), padding="CAUSAL")(h)
        h = einops.rearrange(h, "c l 1 -> l c")
        h = nn.gelu(h)
        h = S6(self.state_dim, self.complex)(h)
        h = nn.Dense(channels, use_bias=False)(h * g)

        return x + h
