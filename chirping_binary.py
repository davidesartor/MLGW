from __future__ import annotations
from typing import NamedTuple
import random
import numpy as np
import numpy.typing as npt


SPEED_LIGHT_SIU = 299792458.0
NEWTON_CONST_SIU = 6.6743e-11
MEGAPARSEC_TO_m = 3.086e22
SOLARMASS_TO_Kg = 1.989e30


class Parameters(NamedTuple):
    r_Mpc: float
    Mc_smass: float
    theta_rad: float
    tcoal_s: float
    phi0_rad: float

    @property
    def r_m(self) -> float:
        return self.r_Mpc * MEGAPARSEC_TO_m

    @property
    def Mc_kg(self) -> float:
        return self.Mc_smass * SOLARMASS_TO_Kg

    @classmethod
    def sample(cls, time_range: tuple[float, float]) -> Parameters:
        # distance uniform in volume (exlude approx solar sistem)
        r_Mpc = 4000 * (random.uniform(1e-11, 1) ** (1 / 3))
        m1_smass = random.uniform(25, 100)
        m2_smass = random.uniform(25, 100)
        Mc_smass = (m1_smass * m2_smass) ** (3 / 5) / (m1_smass + m2_smass) ** (1 / 5)
        theta_rad = random.uniform(0, np.pi)
        tcoal_s = random.uniform(*time_range)
        phi0_rad = random.uniform(0, 2 * np.pi)
        return cls(r_Mpc, Mc_smass, theta_rad, tcoal_s, phi0_rad)


def amplitude(times: npt.NDArray[np.floating], parameters: Parameters) -> npt.NDArray[np.floating]:
    tau = (parameters.tcoal_s - times) * (parameters.tcoal_s > times)
    G_Mc_over_c3 = NEWTON_CONST_SIU * parameters.Mc_kg / SPEED_LIGHT_SIU**3
    amp =  (SPEED_LIGHT_SIU / parameters.r_m) * (G_Mc_over_c3) ** (5 / 4) * (5 / tau) ** (1 / 4)
    return np.where(tau > 0, amp, 0)


def phase(times: npt.NDArray[np.floating], parameters: Parameters) -> npt.NDArray[np.floating]:
    tau = (parameters.tcoal_s - times) * (parameters.tcoal_s > times)
    G_Mc_over_c3 = NEWTON_CONST_SIU * parameters.Mc_kg / SPEED_LIGHT_SIU**3
    return parameters.phi0_rad - 2.0 * (tau / (5 * G_Mc_over_c3)) ** (5 / 8)


def h_cross(times: npt.NDArray[np.floating], parameters: Parameters) -> npt.NDArray[np.floating]:
    amp = amplitude(times, parameters)
    phi = phase(times, parameters)
    return amp * np.cos(parameters.theta_rad) * np.sin(phi)


def h_plus(times: npt.NDArray[np.floating], parameters: Parameters) -> npt.NDArray[np.floating]:
    amp = amplitude(times, parameters)
    phi = phase(times, parameters)
    return amp * (1 + np.cos(parameters.theta_rad) ** 2) / 2 * np.cos(phi)
