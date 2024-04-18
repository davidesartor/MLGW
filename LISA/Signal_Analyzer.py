import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.fft import fft, ifft

class Signal_Analyzer:
    @staticmethod
    def plot_signal(signal_df, column='Signal', title='Signal Plot', ax=None):
        """
        Plots a given signal from a DataFrame.

        Parameters:
        signal_df : pandas.DataFrame
            DataFrame containing the signal to plot, with at least 'Time' and the specified 'column'.
        column : str, optional
            The column name in signal_df that contains the signal values. Default is 'Signal'.
        title : str, optional
            The title of the plot. Default is 'Signal Plot'.
        """
        if column not in signal_df.columns:
            raise ValueError(f"Column '{column}' not found in DataFrame.")

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(signal_df['Time'], signal_df[column])
        ax.set_xlabel('Time')
        ax.set_ylabel('Amplitude')
        ax.set_title(title)
        ax.grid(True)

    @staticmethod
    def plot_fft(signal_df, column='Signal', title='FFT Spectrum', use_abs=True, ax=None):
        """
        Plots the Fast Fourier Transform (FFT) of a given signal, with an option to plot using absolute values.

        Parameters:
        signal_df : pandas.DataFrame
            DataFrame containing the signal to plot, with at least 'Time' and the specified 'column'.
        column : str, optional
            The column name in signal_df that contains the signal values. Default is 'Signal'.
        title : str, optional
            The title of the plot. Default is 'FFT Spectrum'.
        use_abs : bool, optional
            Whether to plot the FFT using absolute values. Default is False.
        """
        if column not in signal_df.columns:
            raise ValueError(f"Column '{column}' not found in DataFrame.")

        signal = signal_df[column].to_numpy()
        sample_rate = 1.0 / (signal_df['Time'].iloc[1] - signal_df['Time'].iloc[0])
        fft_signal = np.fft(signal)
        fft_freqs = np.fft.fftfreq(len(signal), d=1/sample_rate)

        y_values = np.abs(fft_signal) if use_abs else fft_signal

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(fft_freqs[:len(signal)//2], y_values[:len(signal)//2])
        ax.set_xlabel('Frequency')
        ax.set_ylabel('Amplitude' if use_abs else 'Value')
        ax.set_title(title)
        ax.grid(True)