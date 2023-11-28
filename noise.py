from functools import cache
import numpy as np


def generate(dt, n, colour=None, rng=np.random.default_rng()):
    """Generates uniformly sampled noise of a particular colour.
    Parameters
    ----------
    dt : float
        Sampling period, in seconds.
    n : int
        Number of samples to generate.
    colour : function, optional
        Colouring function that specifies the PSD at a given frequency. If
        not specified, the noise returned will be white Gaussian noise with
        a PSD of 1.0 across the entire frequency range.
    Returns
    -------
    numpy.array
        Length `n` array of sampled noise.
    """

    """Produces a frequency domain representation of uniformly sampled
    Gaussian white noise.
    """
    # Calculate random frequency components
    f = np.fft.rfftfreq(n, dt)
    x_f = rng.normal(0, 0.5, len(f)) + 1j * rng.normal(0, 0.5, len(f))
    x_f *= np.sqrt(n / dt)

    # Ensure our 0 Hz and Nyquist components are purely real
    x_f[0] = np.abs(x_f[0])
    if len(f) % 2 == 0:
        x_f[-1] = np.abs(x_f[-1])

    if colour:
        x_f *= np.sqrt(colour(f))
    return np.fft.irfft(x_f)


def piecewise_logarithmic(frequencies, psds):
    """Creates a custom piecewise colouring function
    Parameters
    ----------
    frequencies : numpy.array
        Array of frequencies, in Hz
    psds : numpy.array
        Array of PSDs
    Returns
    -------
    function
        Custom noise colouring function that can be used with `generate()`.
        The function is linearly interpolated in log space for the given
        frequencies and PSDs. Values outside the range of frequencies given
        are set to the PSD of the closest endpoint.
    """
    # Convert to log space
    log_frequencies = np.log(frequencies)
    log_psds = np.log(psds)

    # Create a closure for our colour function that suppresses the warning
    # about np.log(0) (which correctly returns -np.inf anyway)
    def colour(f):
        with np.errstate(divide="ignore"):
            return np.exp(np.interp(np.log(f), log_frequencies, log_psds))

    return colour


@cache
def LIGOL():
    frequencies, psds = np.loadtxt("LIGOLpsd.txt", dtype="double", unpack=True)
    return piecewise_logarithmic(frequencies, psds)


def white(scale=1.0):
    return lambda f: scale


def pink(scale=1.0):
    return lambda f: scale / np.where(f == 0.0, np.inf, f)


def brownian(scale=1.0):
    return lambda f: scale / np.where(f == 0.0, np.inf, f**2)


def blue(scale=1.0):
    return lambda f: scale * f


def violet(scale=1.0):
    return lambda f: scale * f**2


# Some common aliases for the various colours of noise
brown = brownian
red = brownian
azure = blue
purple = violet
