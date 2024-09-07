import argparse
import itertools
import test

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import entropy
from tqdm import tqdm

parser = argparse.ArgumentParser(description="as the file label")
parser.add_argument(
    "--file",
    type=str,
    default="",
    help="The files to read to calculate the entropy and histogram",
)
args = parser.parse_args()
print(args)

df_all = pd.read_csv(f"{args.file}_data.csv")
print(f"=====================READ Data: \n{df_all}\n======================\n")

leg_17_data = df_all[df_all["Leg"] == 17]["Phase"].to_numpy()


def visualize_resampling(original_timestamps, original_phases):
    plt.figure(figsize=(12, 6))
    plt.step(original_timestamps, original_phases, where="post", label="Original Data")
    plt.step(
        new_timestamps,
        resampled_phases,
        where="post",
        linestyle="--",
        label="Resampled Data",
    )
    plt.xlabel("Time (s)")
    plt.ylabel("Phase")
    plt.title("Original vs Resampled Phase Data")
    plt.legend()
    plt.show()


def get_sin_wave_plot(new_timestamps, resampled_phases, freq=0.2):
    """
    This function might have bugs in implmentation.
    """
    fundamental_frequency = freq  # Fundamental frequency of the square wave (Hz)
    approx_wave = np.zeros_like(new_timestamps)

    # Number of harmonics to consider for the approximation
    num_harmonics = 5

    for n in range(1, num_harmonics * 2, 2):  # Loop over odd harmonics (1, 3, 5, ...)
        approx_wave += (
            (4 / np.pi)
            * (1 / n)
            * np.sin(2 * np.pi * n * fundamental_frequency * new_timestamps)
        )

    # Plot the original square wave and the Fourier approximation
    plt.figure(figsize=(10, 6))
    time_length_scale = 1
    plt.plot(
        new_timestamps[: len(new_timestamps) // time_length_scale],
        resampled_phases[: len(resampled_phases) // time_length_scale],
        label="Original Square Wave",
        alpha=0.6,
    )
    plt.plot(
        new_timestamps[: len(new_timestamps) // time_length_scale],
        approx_wave[: len(approx_wave) // time_length_scale],
        label="Fourier Approximation ({} Harmonics)".format(num_harmonics),
        linestyle="--",
    )
    plt.title("Square Wave vs Fourier Series Approximation")
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.show()

    return approx_wave


def perform_fft_analysis(
    new_timestamps, resampled_phases, top_n_frequencies=5, figure=False
):
    """
    Performs FFT analysis on the resampled phase data and plots the frequency spectrum.

    Parameters:
        new_timestamps (np.array): Array of time values corresponding to the resampled data.
        resampled_phases (np.array): Array of phase data to analyze (binary square wave).
        top_n_frequencies (int): Number of top dominant frequencies to identify.

    Returns:
        dominant_frequencies (np.array): Array of the top N dominant frequencies by amplitude.
    """
    # Compute the average time interval (dt) and sampling frequency (Fs)
    dt = np.mean(np.diff(new_timestamps))  # Average time interval between samples
    Fs = 1.0 / dt  # Sampling frequency (Hz)

    # Perform FFT on the resampled phase data
    fft_result = np.fft.fft(resampled_phases)
    N = len(fft_result)  # Number of samples

    # Generate the corresponding frequency bins
    frequencies = np.fft.fftfreq(N, d=dt)

    # Get the amplitude of the FFT (absolute value and normalized by N)
    fft_amplitude = np.abs(fft_result) / N

    # Only keep the positive frequencies (since FFT output is symmetric)
    positive_freq_indices = frequencies > 0
    frequencies = frequencies[positive_freq_indices]
    fft_amplitude = fft_amplitude[positive_freq_indices]

    # Plot the FFT result (Frequency spectrum)
    if figure:
        plt.figure(figsize=(10, 6))
        plt.plot(frequencies, fft_amplitude)
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Amplitude")
        plt.title("Frequency Spectrum of Resampled Phase Data")
        plt.show()

    # Identify the top dominant frequencies (by amplitude)
    dominant_frequencies = frequencies[np.argsort(fft_amplitude)[-top_n_frequencies:]]
    return dominant_frequencies


if __name__ == "__main__":
    print("================FFT Analysis:")
    original_timestamps = df_all[df_all["Leg"] == 18]["Timestamp"].to_numpy()
    original_phases = df_all[df_all["Leg"] == 18]["Phase"].to_numpy()
    assert len(original_timestamps) == len(
        original_phases
    ), "Timestamp and Phases data length mismatch"

    print(len(original_timestamps), len(original_phases))
    verbose = False
    if verbose:
        np.set_printoptions(threshold=np.inf)
        print(original_phases)
        print(original_timestamps)

    high_freq_interval = np.mean(np.diff(original_timestamps)) / 10  # 1  # 8  # 10
    new_timestamps = np.arange(
        original_timestamps[0], original_timestamps[-1], high_freq_interval
    )
    resampled_phases = np.interp(new_timestamps, original_timestamps, original_phases)
    if verbose:
        visualize_resampling(original_timestamps, original_phases)

    print(len(new_timestamps), len(resampled_phases))
    if verbose:
        print(resampled_phases)
        print(new_timestamps)

    # FFT Analysis
    # @ref:
    # https://medium.com/ai-does-it-better/numpy-scipy-ffts-distinct-performance-real-valued-optimizations-917a9f649aa5
    dominant_frequencies = perform_fft_analysis(
        new_timestamps, resampled_phases, top_n_frequencies=5, figure=True
    )
    print("Top 5 Dominant Frequencies: ", dominant_frequencies)
    print()

    # TODO:
    # 1. stance-swing phase span histogram (√)
    # 2. dominant frequency overtime (√)
    # 3. convert to sin

    window_size = 100

    print("FFT Analysis on Sliding Window:")
    dominant_frequencies_overtime = []
    window_size_freq = window_size * 10
    for i in tqdm(
        range(len(resampled_phases) - window_size_freq + 1),
        desc="Calculating FFT over-time",
    ):
        window_data = resampled_phases[i : i + window_size_freq]
        window_timestamps = new_timestamps[i : i + window_size_freq]
        dominant_frequencies = perform_fft_analysis(
            window_timestamps, window_data, top_n_frequencies=5
        )
        dominant_frequencies_overtime.append(dominant_frequencies[0])

    plt.figure(figsize=(12, 8))
    x_axis = np.arange(len(dominant_frequencies_overtime))
    plt.plot(x_axis, dominant_frequencies_overtime, label="Top Frequency Over Time")
    plt.title("Sliding Window FFT Analysis")
    plt.xlabel("Window Index")
    plt.ylabel("Dominant Frequencies (Hz)")
    plt.legend()
    plt.show()

    print("For the full window, get sin wave")
    sin_wave = get_sin_wave_plot(
        new_timestamps, resampled_phases, dominant_frequencies[0]
    )
    print("Sin Wave: ", len(sin_wave))
    print("Square Wave: ", len(resampled_phases))

    dominant_frequencies = perform_fft_analysis(
        new_timestamps, sin_wave, top_n_frequencies=5, figure=True
    )
    print("Top 5 Dominant Frequencies For Sin: ", dominant_frequencies)
    print()
