import argparse
import itertools
import test

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import entropy
from tqdm import tqdm

gif_front_swing_prob = None
gif_middle_swing_prob = None
gif_hind_swing_prob = None
gif_front_swing_prob_2 = None
gif_middle_swing_prob_2 = None
gif_hind_swing_prob_2 = None

gif_front_swing_prob = np.load("gif_front_swing_prob.npy")
gif_middle_swing_prob = np.load("gif_middle_swing_prob.npy")
gif_hind_swing_prob = np.load("gif_hind_swing_prob.npy")
gif_front_swing_prob_2 = np.load("gif_front_swing_prob_2.npy")
gif_middle_swing_prob_2 = np.load("gif_middle_swing_prob_2.npy")
gif_hind_swing_prob_2 = np.load("gif_hind_swing_prob_2.npy")


def visualize_resampling(
    original_timestamps, original_phases, new_timestamps, resampled_phases
):
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
    dt = np.mean(np.diff(new_timestamps))  # Average time interval between samples
    Fs = 1.0 / dt  # Sampling frequency (Hz)

    fft_result = np.fft.fft(resampled_phases)
    N = len(fft_result)

    frequencies = np.fft.fftfreq(N, d=dt)
    fft_amplitude = np.abs(fft_result) / N

    positive_freq_indices = frequencies > 0
    frequencies = frequencies[positive_freq_indices]
    fft_amplitude = fft_amplitude[positive_freq_indices]

    if figure:
        plt.figure(figsize=(10, 6))
        plt.plot(frequencies, fft_amplitude)
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Amplitude")
        plt.title("Frequency Spectrum of Resampled Phase Data")
        plt.show()

    dominant_frequencies = frequencies[np.argsort(fft_amplitude)[-top_n_frequencies:]]
    return dominant_frequencies


# Do FFT Analysis on the swing probabilities
