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

    @staticmethod
    def combine_signals(signal_df1, signal_df2, column1='Signal', column2='Signal'):
        """
        Combines two signals from two DataFrames and returns a new DataFrame with the combined signal.

        Parameters:
        signal_df1 : pandas.DataFrame
            DataFrame containing the first signal to combine, with at least 'Time' and the specified 'column1'.
        signal_df2 : pandas.DataFrame
            DataFrame containing the second signal to combine, with at least 'Time' and the specified 'column2'.
        column1 : str, optional
            The column name in signal_df1 that contains the first signal values. Default is 'Signal'.
        column2 : str, optional
            The column name in signal_df2 that contains the second signal values. Default is 'Signal'.

        Returns:
        combined_signal_df : pandas.DataFrame
            A DataFrame containing the combined signal, with columns 'Time' and 'Combined_Signal'.
        """
        if column1 not in signal_df1.columns or column2 not in signal_df2.columns:
            raise ValueError("Column not found in one or both DataFrames.")

        # Ensure the time axes are aligned
        if not np.array_equal(signal_df1['Time'], signal_df2['Time']):
            raise ValueError("Time axes of the two signals are not aligned.")

        combined_signal = signal_df1[column1] + signal_df2[column2]
        combined_signal_df = signal_df1[['Time']].copy()
        combined_signal_df['Combined_Signal'] = combined_signal

        return combined_signal_df