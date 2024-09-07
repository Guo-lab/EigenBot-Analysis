import itertools
from itertools import islice

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy.stats import entropy, wasserstein_distance
from tqdm import tqdm


def is_sublist(sublist, data_list):
    sublist_len = len(sublist)
    return any(
        sublist == list(islice(data_list, i, i + sublist_len))
        for i in range(len(data_list) - sublist_len + 1)
    )


def get_cycle_span_to_series(data):
    change_points = np.where(np.diff(data) != 0)[0] + 1
    # print("Change points: ", change_points)

    groups = np.split(data, change_points)
    phase_labels = np.array([group[0] for group in groups])
    lengths = np.array([len(group) for group in groups])

    print("Lengths: ", lengths)
    return phase_labels, lengths


def get_cycle_span_to_series_test():
    print("\nTest:")
    data = np.array([0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1])
    phase_labels, lengths = get_cycle_span_to_series(data)
    print("Phase Labels:", phase_labels)
    print("Lengths:", lengths)
    print()


def sin_wave_test():
    t = np.linspace(0, 1, 500, endpoint=False)  # Resampled time vector
    square_wave = np.sign(np.sin(2 * np.pi * 5 * t))  # A square wave of frequency 5 Hz

    fundamental_frequency = 5  # Fundamental frequency of the square wave (Hz)
    approx_wave = np.zeros_like(t)
    num_harmonics = 5
    for n in range(1, num_harmonics * 2, 2):  # Loop over odd harmonics (1, 3, 5, ...)
        approx_wave += (
            (4 / np.pi) * (1 / n) * np.sin(2 * np.pi * n * fundamental_frequency * t)
        )

    plt.figure(figsize=(10, 6))
    plt.plot(t, square_wave, label="Original Square Wave", alpha=0.6)
    plt.plot(
        t,
        approx_wave,
        label="Fourier Approximation ({} Harmonics)".format(num_harmonics),
        linestyle="--",
    )
    plt.title("Square Wave vs Fourier Series Approximation")
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.show()


def sub_list_function_test():
    print("\nTest:")
    data = [0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1]
    print(is_sublist([1, 1, 0], data))
    print(is_sublist([1, 1, 1, 1, 0], data))
    print(is_sublist([0, 0, 0], data))
    print(is_sublist([0, 0, 1], data))
    print()


def double_hill_test():
    time_bins = np.array([1, 2, 3, 4, 5])
    swing_counts_front = [0.2, 0.4, 0.3, 0.5, 0.6]
    swing_counts_middle = [0.1, 0.3, 0.4, 0.2, 0.5]
    swing_counts_hind = [0.3, 0.2, 0.5, 0.4, 0.1]

    bar_width = 0.06
    positions_front = time_bins - bar_width
    positions_hind = time_bins
    fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    axs[0].bar(
        positions_front,
        swing_counts_front,
        width=bar_width,
        color="blue",
        alpha=0.4,
        label="Right Front Leg",
    )
    axs[0].bar(
        positions_hind,
        swing_counts_hind,
        width=bar_width,
        color="purple",
        alpha=0.4,
        label="Right Hind Leg",
    )
    axs[0].set_title("Swing Phase Frequency - Right Front Leg. Right Hind Leg")
    axs[0].set_ylabel("Swing Probability")
    axs[0].legend()

    axs[1].bar(
        time_bins,
        swing_counts_middle,
        width=0.06,
        color="red",
        alpha=0.4,
        label="Right Middle Leg",
    )
    axs[1].set_title("Swing Phase Frequency - Right Middle Leg")
    axs[1].set_ylabel("Swing Probability")
    axs[1].legend()

    plt.tight_layout()
    plt.show()


def upper_case_test():
    lowercase = "abcdefghijklmnopqrstuvwxyz"
    uppercase = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    return lowercase.capitalize(), uppercase.capitalize()


def KL(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    print(a, b)
    return np.sum(np.where(a != 0, a * np.log(a / b), 0))


def KL_divergence(p, q):
    print(p, q)
    p[p == 0] = 1e-6
    q[q == 0] = 1e-6
    vec = scipy.special.rel_entr(p, q)
    kl_div = np.sum(vec)
    return kl_div


def KL_divergence_test():
    p = [4, 7, 12, 3]
    q = [3, 12, 11, 2]

    print(KL(p, q))
    print(KL_divergence(p, q))
    assert KL(p, q) == KL_divergence(p, q), "KL Test failed"

    print(KL(p / np.sum(p), q / np.sum(q)))
    print(KL(q / np.sum(q), p / np.sum(p)))

    print(KL_divergence(p / np.sum(p), q / np.sum(q)))
    print(KL_divergence(q / np.sum(q), p / np.sum(p)))


def KL_divergence_real_test():
    a = [
        0.07692308,
        0.0,
        0.07692308,
        0.0,
        0.0,
        0.07692308,
        0.0,
        0.15384615,
    ]
    b = [
        0.0,
        0.0,
        0.07692308,
        0.0,
        0.0,
        0.07692308,
        0.0,
        0.07692308,
    ]
    print(np.sum(a))
    print(np.sum(b))
    print(KL_divergence(a, b))


def wasserstein_distance_test():
    X1 = np.array([6, 1, 2, 3, 5, 5, 1])
    X2 = np.array([1, 4, 3, 1, 6, 6, 4])
    print(wasserstein_distance(X1, X2))
    print(wasserstein_distance(X2, X1))


def test_msc():
    # Generate example signals
    fs = 500  # Sampling frequency
    N = 1000  # Number of points
    t = np.arange(N) / fs

    # Generate a 10 Hz square wave
    x = signal.square(2 * np.pi * 10 * t)  # Square wave 1 (10 Hz)

    # Generate a noisy square wave without a clear pattern
    # Introduce random phase shifts to make the wave less clear
    random_phase = np.cumsum(np.random.randn(N) * 0.1)  # Random phase variations
    y = signal.square(2 * np.pi * 10 * t + random_phase)  # Square wave with noise

    # Calculate magnitude-squared coherence
    f, Cxy = signal.coherence(x, y, fs, nperseg=256)

    # Plot the two square waves
    plt.figure(figsize=(10, 6))

    # Subplot 1: Square waves
    plt.subplot(2, 1, 1)
    plt.plot(t, x, label="Square Wave 1 (10 Hz)")
    plt.plot(t, y, label="Square Wave 2 (Randomized Phase)", alpha=0.7)
    plt.title("Square Waves (with and without clear pattern)")
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.legend()

    # Subplot 2: Magnitude-Squared Coherence
    plt.subplot(2, 1, 2)
    plt.semilogy(f, Cxy)
    plt.title("Magnitude-Squared Coherence (Square Waves)")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Coherence")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # sin_wave_test()
    # sub_list_function_test()
    # double_hill_test()

    # print(upper_case_test())
    KL_divergence_test()
    wasserstein_distance_test()
    KL_divergence_real_test()
    test_msc()
