import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft

class Clean_Signal_Generator:
    def __init__(self, total_time=1000, num_sources=100, mean_amp=10, sigma_amp=2, freq_range=(1e-4, 1e-1),
                 amp_distribution_func=None, omega_distribution_func=None):
        """
        Initializes the Clean_Signal_Generator class with given parameters.

        Parameters:
        total_time : float
            | The total time duration of the signal.
        num_sources : int
            | The number of sinusoidal components in the signal.
        mean_amp : float
            | The mean amplitude of the sinusoidal components.
        sigma_amp : float
            | The standard deviation of the amplitudes of the sinusoidal components.
        freq_range : tuple of (float, float)
            | A tuple containing the minimum and maximum frequencies for the sinusoidal components.
        amp_distribution_func : callable, optional
            | A function to generate amplitudes for the sinusoidal components.
        omega_distribution_func : callable, optional
            | A function to generate angular frequencies for the sinusoidal components.
        """
        self.total_time = total_time
        self.num_sources = num_sources
        self.mean_amp = mean_amp
        self.sigma_amp = sigma_amp
        self.min_freq, self.max_freq = freq_range
        self.amp_distribution_func = amp_distribution_func if amp_distribution_func else self.default_amp_distribution
        self.omega_distribution_func = omega_distribution_func if omega_distribution_func else self.default_omega_distribution
        self.A = None
        self.omega = None
        self.theta = None
        self.generated_signal = None

    def default_amp_distribution(self, size):
        """
        Default amplitude distribution using normal distribution.
        """
        return np.random.normal(loc=self.mean_amp, scale=self.sigma_amp, size=size)

    def default_omega_distribution(self, size):
        """
        Default angular frequency distribution using normal distribution.
        """
        mean_freq = (self.min_freq + self.max_freq) / 2
        sigma_freq = (self.max_freq - self.min_freq) / 6
        return np.random.normal(loc=2 * np.pi * mean_freq, scale=2 * np.pi * sigma_freq, size=size)

    def generating_signal(self, amp_uselog=False):
        """
        Generates a clean signal composed of multiple sinusoidal components and returns it as a DataFrame.

        Parameters:
        amp_uselog : bool, optional
            | If True, the amplitude values are sampled from a log-normal distribution.
            | If False, they are sampled from a normal distribution. Default is False.

        Returns:
        A DataFrame containing the generated signal, with columns 'Time' and 'Clean_Signal'.
        """
        time_interval = 0.05 / self.max_freq
        num_samples = int(self.total_time / time_interval)
        t = np.linspace(0, self.total_time, num_samples, endpoint=False)
        d_t = np.zeros(num_samples)

        if amp_uselog:
            mean_logamp = np.log(self.mean_amp)
            sigma_logamp = np.log(self.sigma_amp)
            lnA = np.random.normal(loc=mean_logamp, scale=sigma_logamp, size=self.num_sources)
            self.A = np.exp(lnA)
        else:
            self.A = self.amp_distribution_func(self.num_sources)

        self.omega = self.omega_distribution_func(self.num_sources)
        self.theta = np.random.uniform(0, 2 * np.pi, size=self.num_sources)

        for i in range(self.num_sources):
            d_t += self.A[i] * np.sin(self.omega[i] * t + self.theta[i])

        self.generated_signal = pd.DataFrame({'Time': t, 'Signal': d_t})
        return self.generated_signal

    def plot_individual_components(self, num_components=5):
        """
        Plots individual sinusoidal components of the generated signal.

        Parameters:
        num_components : int, optional
            | The number of components to plot. Default is 5.
        """
        if self.generated_signal is not None:
            plt.figure(figsize=(10, 6))
            t = self.generated_signal['Time']
            for i in range(min(num_components, self.num_sources)):
                component_signal = self.A[i] * np.sin(self.omega[i] * t + self.theta[i])
                plt.plot(t, component_signal, label=f'Component {i + 1}')
            plt.xlabel('Time')
            plt.ylabel('Signal')
            plt.title(f'Individual Components (Up to {num_components})')
            plt.legend()
            plt.grid(True)
            plt.show()
        else:
            print("Please generate the signal first.")


class Noise_Generator:
    def __init__(self, total_time=1000, psd_file=None, freq_range=(1e-4, 1e-1), noise_amplitude=10):
        """
        Initializes the Noise_Generator class with given parameters.

        Parameters:
        total_time : float
            The total time duration for the noise signal.
        psd_file : str, optional
            Path to the PSD file used for generating noise based on power spectral density.
        default_freq_range : tuple of (float, float), optional
            A tuple containing the default minimum and maximum frequencies for the noise components.
        noise_amplitude : float, optional
            The amplitude scaling factor for the generated noise.
        """
        self.total_time = total_time
        self.psd_file = psd_file
        self.freq_range = freq_range
        self.noise_amplitude = noise_amplitude

    def read_psd_data(self):
        """
        Reads the power spectral density data from the PSD file and returns it.

        Returns:
        psd_freqs, psd_amps : numpy arrays
            Arrays containing the frequencies and corresponding amplitudes from the PSD data.
        """
        try:
            psd_data = np.loadtxt(self.psd_file)
            psd_freqs = psd_data[:, 0]
            psd_amps = psd_data[:, 1]
            return psd_freqs, psd_amps
        except IOError as e:
            print(f"Error reading PSD file: {e}")
            return None, None

    def plot_psd(self):
        """
        Plots the Power Spectral Density (PSD) data from the PSD file.
        """
        psd_freqs, psd_amps = self.read_psd_data()

        if psd_freqs is not None and psd_amps is not None:
            plt.figure(figsize=(10, 6))
            plt.plot(psd_freqs, psd_amps)
            plt.xscale('log')
            plt.yscale('log')
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('Amplitude')
            plt.title('Power Spectral Density (Log-scale)')
            plt.grid(True)
            plt.show()
        else:
            print("PSD data not available or unable to read PSD file.")
            
    def get_freq_range(self):
        """
        Retrieves the frequency range either from the PSD file or the default frequency range.

        Returns:
        freq_range : tuple of (float, float)
            A tuple containing the minimum and maximum frequencies for the noise components.
        """
        if self.psd_file:
            psd_freqs, _ = self.read_psd_data()
            if psd_freqs is not None:
                return (np.min(psd_freqs), np.max(psd_freqs))
            else:
                print("Unable to read PSD file, using default frequency range.")
        return self.freq_range

    def generate_noise(self, noise_type='white'):
        """
        Generates noise based on the specified type ('white' or 'psd') and returns it as a DataFrame.

        Parameters:
        noise_type : str, optional
            The type of noise to generate. Options are 'white' for white noise and 'psd' for noise based on a PSD file.

        Returns:
        noise_signal_df : pandas.DataFrame
            A DataFrame containing the generated noise signal, with columns 'Time' and 'Noise'.
        """
        time_interval = 0.05 / self.freq_range[1]  # Ensure at least 20 samples per highest frequency component
        num_samples = int(self.total_time / time_interval)
        t = np.linspace(0, self.total_time, num_samples, endpoint=False)

        if noise_type == 'white':
            noise = np.random.normal(scale=self.noise_amplitude, size=num_samples)
        elif noise_type == 'psd' and self.psd_file:
            psd_freqs, psd_amps = self.read_psd_data()
            sample_rate = 1.0 / time_interval  # Calculate the sample rate
            freqs = np.fft.fftfreq(num_samples, d=1 / sample_rate)
            
            noise_psd = np.interp(freqs, psd_freqs, psd_amps, left=0, right=0)
            noise_psd_fft = np.sqrt(noise_psd)
            noise = np.fft.ifft(noise_psd_fft).real

        noise_df = pd.DataFrame({'Time': t, 'Signal': noise})
        return noise_df