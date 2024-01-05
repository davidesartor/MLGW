from dataclasses import dataclass
from typing import NamedTuple
import jax
import jax.numpy as jnp
from . import noise

SPEED_LIGHT_SIU = 299792458.0
NEWTON_CONST_SIU = 6.6743e-11
MEGAPARSEC_TO_m = 3.086e22
SOLARMASS_TO_Kg = 1.989e30


class Parameters(NamedTuple):
    r_Mpc: jax.Array
    Mc_smass: jax.Array
    iota_rad: jax.Array
    tcoal_s: jax.Array
    phi0_rad: jax.Array

    @property
    def r_m(self) -> jax.Array:
        return self.r_Mpc * MEGAPARSEC_TO_m

    @property
    def Mc_kg(self) -> jax.Array:
        return self.Mc_smass * SOLARMASS_TO_Kg


def sample_params(rng_key: jax.Array, time_range: tuple[float, float]) -> Parameters:
    rng_subkeys = jax.random.split(rng_key, 6)
    # distance uniform in volume (exlude radius 1e-11 Gpc = approx solar sistem radius)
    r_Mpc = 4 * (jax.random.uniform(rng_subkeys[0], minval=1e-10) ** (1 / 3))
    # masses uniform in (25, 100) solar masses
    m1_smass = jax.random.uniform(rng_subkeys[1], minval=25, maxval=100)
    m2_smass = jax.random.uniform(rng_subkeys[2], minval=25, maxval=100)
    Mc_smass = (m1_smass * m2_smass) ** (3 / 5) / (m1_smass + m2_smass) ** (1 / 5)
    # angles uniform in sphere
    iota_rad = jax.random.uniform(rng_subkeys[3], maxval=jnp.pi)
    phi0_rad = jax.random.uniform(rng_subkeys[4], maxval=2 * jnp.pi)
    # coalescence time uniform in time_range
    tcoal_s = jax.random.uniform(rng_subkeys[5], minval=time_range[0], maxval=time_range[1])
    return Parameters(r_Mpc, Mc_smass, iota_rad, tcoal_s, phi0_rad)


def amplitude(params: Parameters, times: jax.Array, scale=1.0) -> jax.Array:
    tau = (params.tcoal_s - times) * (params.tcoal_s > times)
    G_Mc_over_c3 = NEWTON_CONST_SIU * params.Mc_kg / SPEED_LIGHT_SIU**3
    amp = (SPEED_LIGHT_SIU / params.r_m) * (G_Mc_over_c3) ** (5 / 4) * (5 / tau) ** (1 / 4)
    return jnp.where(tau > 0, scale * amp, 0)


def phase(params: Parameters, times: jax.Array) -> jax.Array:
    tau = (params.tcoal_s - times) * (params.tcoal_s > times)
    G_Mc_over_c3 = NEWTON_CONST_SIU * params.Mc_kg / SPEED_LIGHT_SIU**3
    return params.phi0_rad - 2.0 * (tau / (5 * G_Mc_over_c3)) ** (5 / 8)


def h_cross(params: Parameters, times: jax.Array, scale=1.0) -> jax.Array:
    amp = amplitude(params, times, scale)
    phi = phase(params, times)
    return amp * jnp.cos(params.iota_rad) * jnp.sin(phi)


def h_plus(params: Parameters, times: jax.Array, scale=1.0) -> jax.Array:
    amp = amplitude(params, times, scale)
    phi = phase(params, times)
    return amp * (1 + jnp.cos(params.iota_rad) ** 2) / 2 * jnp.cos(phi)


@dataclass
class ChirpingBinary:
    n_sources: int = 1
    episode_duration_s: float = 10.0
    sample_rate_Hz: float = 10000.0
    scale: float = 1.0

    @property
    def times(self):
        dt = 1 / self.sample_rate_Hz
        return jnp.arange(dt, self.episode_duration_s + dt, dt)

    def sample(self, rng_key: jax.Array):
        dt = 1 / self.sample_rate_Hz

        def get_source(rng_key: jax.Array) -> tuple[Parameters, jax.Array]:
            params = sample_params(rng_key, time_range=(0, self.episode_duration_s))
            signal = h_plus(params, self.times, self.scale)
            return params, signal

        params, source_signal = jax.vmap(get_source)(jax.random.split(rng_key, self.n_sources))
        clean_signal = source_signal.sum(axis=0)

        noise_signal = noise.generate(rng_key, dt, len(self.times), noise.LIGOL(self.scale))
        noisy_signal = clean_signal + noise_signal

        return noisy_signal, clean_signal

    def get_batch(self, rng_key: jax.Array, batch_size: int):
        rng_keys = jax.random.split(rng_key, batch_size)
        clean_signals, noisy_signals = jax.vmap(self.sample)(rng_keys)
        return noisy_signals, clean_signals
