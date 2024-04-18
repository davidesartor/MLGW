import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft

class Signal_Generator:
    def __init__(self, total_time=1000, num_sources=100, mean_amp=10, sigma_amp=2, freq_range=(1e-4, 1e-1),
                 amp_distribution_func=None, omega_distribution_func=None, noise_amplitude=10):
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

        self.noise_amplitude = noise_amplitude

        self.clean_signal = None
        self.noise = None
        self.signal = None

    def _setup_time_array(self):
        """
        Sets up the time array for signal generation.
        """
        time_interval = 0.05 / self.max_freq
        num_samples = int(self.total_time / time_interval)
        t = np.linspace(0, self.total_time, num_samples, endpoint=False)
        return t, num_samples

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

    def generating_clean_signal(self, amp_uselog=False):
        """
        Generates a clean signal composed of multiple sinusoidal components and returns it as a DataFrame.

        Parameters:
        amp_uselog : bool, optional
            | If True, the amplitude values are sampled from a log-normal distribution.
            | If False, they are sampled from a normal distribution. Default is False.

        Returns:
        A DataFrame containing the generated signal, with columns 'Time' and 'Clean_Signal'.
        """
        t, num_samples = self._setup_time_array()
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

        self.clean_signal = pd.DataFrame({'Time': t, 'Signal': d_t})
        return self.clean_signal

    def plot_individual_components(self, num_components=5):
        """
        Plots individual sinusoidal components of the generated signal.

        Parameters:
        num_components : int, optional
            | The number of components to plot. Default is 5.
        """
        if self.clean_signal is not None:
            plt.figure(figsize=(10, 6))
            t = self.clean_signal['Time']
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

    def generating_noise(self):
        """
        Generates noise based on the specified type ('white' or 'psd') and returns it as a DataFrame.

        Parameters:
        noise_type : str, optional
            The type of noise to generate. Options are 'white' for white noise and 'psd' for noise based on a PSD file.

        Returns:
        noise_signal_df : pandas.DataFrame
            A DataFrame containing the generated noise signal, with columns 'Time' and 'Noise'.
        """
        t, num_samples = self._setup_time_array()

        noise = np.random.normal(scale=self.noise_amplitude, size=num_samples)

        self.noise = pd.DataFrame({'Time': t, 'Signal': noise})
        return self.noise
    
    def generating_signal(self):
        # Generate clean signal and noise if they haven't been generated yet
        if self.clean_signal is None:
            self.generating_clean_signal()
        if self.noise is None:
            self.generating_noise()

        # Combine the clean signal and noise
        combined_signal = self.clean_signal['Signal'] + self.noise['Signal']
        self.signal = pd.DataFrame({
            'Time': self.clean_signal['Time'],
            'Clean_Signal': self.clean_signal['Signal'],
            'Noise': self.noise['Signal'],
            'Signal': combined_signal
        })
        return self.signal
    
    def printing_parameters(self):
        """
        Combines and prints all parameter data (amplitude, frequency, phase) into a single list called 'params'.
        
        Returns:
        params: list
            A list containing the parameters of the signal.
        """
        if self.A is not None and self.omega is not None and self.theta is not None:
            # Combining all parameters into one list
            params = self.A.tolist() + self.omega.tolist() + self.theta.tolist()
        else:
            params = []
            print("Some parameters have not been generated yet.")
        
        return params
