import argparse
import itertools
import math
import test

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal
from scipy.signal import find_peaks
from scipy.stats import entropy
from tqdm import tqdm


class Leg:
    def __init__(self, leg, module, phase, timestamp=None):
        self.leg_name = leg
        self.leg_number = module
        self.phase = phase
        self.timestamp = timestamp
        self.resampled_timestamp = None
        self.resampled_phase = None

    def __str__(self):
        return f"Leg: {self.leg_name}, Module #{self.leg_number}, Phase: {self.phase}, Timestamp: {self.timestamp}"

    def __repr__(self):
        return f"Leg: {self.leg_name}, Module #{self.leg_number}, Phase: {self.phase}, Timestamp: {self.timestamp}"

    def resample_phase_based_on_timestamp(self, verbose=False):
        high_freq_interval = np.mean(np.diff(self.timestamp)) / 10
        new_timestamps = np.arange(
            self.timestamp[0], self.timestamp[-1], high_freq_interval
        )
        resampled_phases = np.interp(new_timestamps, self.timestamp, self.phase)
        if verbose:
            visualize_resampling(
                self.timestamp, self.phase, new_timestamps, resampled_phases
            )
        self.resampled_timestamp = new_timestamps
        self.resampled_phase = resampled_phases

    def resample_phase_based_on_common_time_grid(self, common_time_grid, verbose=False):
        resampled_phases = np.interp(
            common_time_grid, self.resampled_timestamp, self.resampled_phase
        )
        if verbose:
            visualize_resampling(
                self.resampled_timestamp,
                self.resampled_phase,
                common_time_grid,
                resampled_phases,
            )
        self.resampled_timestamp = common_time_grid
        self.resampled_phase = resampled_phases
        print(f"len: {len(self.resampled_timestamp)}, {len(self.resampled_phase)}")


parser = argparse.ArgumentParser(description="as the file label")
parser.add_argument(
    "--file",
    type=str,
    default="",
    help="The files to read to calculate the entropy and histogram",
)
args = parser.parse_args()
df_all = pd.read_csv(f"{args.file}_data.csv")
if (df_all["Leg"] == 13).any():
    leg_rf = Leg(
        "RF",
        13,
        df_all[df_all["Leg"] == 13]["Phase"].to_numpy(),
        df_all[df_all["Leg"] == 13]["Timestamp"].to_numpy(),
    )
else:
    leg_rf = None

if (df_all["Leg"] == 5).any():
    leg_lf = Leg(
        "LF",
        5,
        df_all[df_all["Leg"] == 5]["Phase"].to_numpy(),
        df_all[df_all["Leg"] == 5]["Timestamp"].to_numpy(),
    )
else:
    leg_lf = None

if (df_all["Leg"] == 19).any():
    leg_rm = Leg(
        "RM",
        19,
        df_all[df_all["Leg"] == 19]["Phase"].to_numpy(),
        df_all[df_all["Leg"] == 19]["Timestamp"].to_numpy(),
    )
else:
    leg_rm = None

if (df_all["Leg"] == 2).any():
    leg_lm = Leg(
        "LM",
        2,
        df_all[df_all["Leg"] == 2]["Phase"].to_numpy(),
        df_all[df_all["Leg"] == 2]["Timestamp"].to_numpy(),
    )
else:
    leg_lm = None

if (df_all["Leg"] == 17).any():
    leg_rh = Leg(
        "RH",
        17,
        df_all[df_all["Leg"] == 17]["Phase"].to_numpy(),
        df_all[df_all["Leg"] == 17]["Timestamp"].to_numpy(),
    )
else:
    leg_rh = None

if (df_all["Leg"] == 18).any():
    leg_lh = Leg(
        "LH",
        18,
        df_all[df_all["Leg"] == 18]["Phase"].to_numpy(),
        df_all[df_all["Leg"] == 18]["Timestamp"].to_numpy(),
    )
else:
    leg_lh = None


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
    dt = np.mean(np.diff(new_timestamps))  # (dt) Average time interval between samples
    Fs = 1.0 / dt  # (Fs) Sampling frequency: (Hz)

    fft_result = np.fft.fft(resampled_phases)  # Perform FFT on the resampled phase data
    N = len(fft_result)  # Number of samples

    frequencies = np.fft.fftfreq(N, d=dt)  # Generate the corresponding frequency bins
    fft_amplitude = np.abs(fft_result) / N  # Get the amplitude of the FFT

    # Only keep the positive frequencies (since FFT output is symmetric)
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


def get_common_freq(leg_rf, leg_rm, leg_rh, leg_lf, leg_lm, leg_lh):

    legs = [leg_rf, leg_rm, leg_rh, leg_lf, leg_lm, leg_lh]
    valid_legs = [
        leg for leg in legs if leg is not None and leg.resampled_timestamp is not None
    ]

    common_dt = min(np.diff(leg.resampled_timestamp).min() for leg in valid_legs)
    # common_dt = min(
    #     np.diff(leg_rf.resampled_timestamp).min(),
    #     np.diff(leg_lf.resampled_timestamp).min(),
    #     np.diff(leg_rm.resampled_timestamp).min(),
    #     np.diff(leg_lm.resampled_timestamp).min(),
    #     np.diff(leg_rh.resampled_timestamp).min(),
    #     np.diff(leg_lh.resampled_timestamp).min(),
    # )
    common_time_grid = np.arange(
        max(leg.resampled_timestamp[0] for leg in valid_legs),
        min(leg.resampled_timestamp[-1] for leg in valid_legs),
        common_dt,
    )

    return 1 / common_dt, common_time_grid  # fs, time_grid (final timestamp)


def calculate_magnitude_squared_coherence(leg1, leg2, fs, nperseg=256):
    f, Cxy = signal.coherence(
        leg1.resampled_phase, leg2.resampled_phase, fs, nperseg=nperseg
    )
    return f, Cxy


def plot_msc(f_list, Cxy_list, tag_list):
    fig, axs = plt.subplots(1, 5, figsize=(12, 3))

    for i, (f, Cxy) in enumerate(zip(f_list, Cxy_list)):
        axs[i].semilogy(f, Cxy)
        axs[i].set_title(f"Coherence: {tag_list[i]}")
        axs[i].set_xlabel("Frequency [Hz]")
        axs[i].set_ylabel("Coherence")

    plt.tight_layout()
    plt.show()


def plot_msc_with_peaks(f_list, Cxy_list, tag_list, peaks_list, top_freq_indices):
    fig, axs = plt.subplots(1, 5, figsize=(12, 3))
    print(len(top_freq_indices))
    for i, (f, Cxy, peaks, top_freq_index) in enumerate(
        zip(f_list, Cxy_list, peaks_list, top_freq_indices)
    ):
        axs[i].semilogy(f, Cxy)
        axs[i].plot(f[peaks], Cxy[peaks], "x", markersize=3)
        top_freq_real_idx = peaks[top_freq_index]
        # print(f"Top Dominant Frequencies Index: {top_freq_real_idx}")
        # print(f"Top Freq's Coherence: {Cxy[top_freq_real_idx]}")
        print(
            f"Top Dominant Frequencies: {[math.floor(x) for x in f[top_freq_real_idx]]}"
        )
        axs[i].plot(
            f[top_freq_real_idx], Cxy[top_freq_real_idx], "o", color="red", markersize=5
        )

        axs[i].set_title(f"Coherence: {tag_list[i]}")
        axs[i].set_xlabel("Frequency [Hz]")
        axs[i].set_ylabel("Coherence")

    plt.tight_layout()
    plt.show()
    # exit()


def get_phase_diff_between_pair_of_legs(leg1, leg2, fs, nperseg=256):
    f, Pxy = signal.csd(leg1.resampled_phase, leg2.resampled_phase, fs, nperseg=nperseg)
    phase_diff = np.angle(Pxy)
    return f, phase_diff


def plot_phase_diff(f_list, phase_diff_list, tag_list):
    fig, axs = plt.subplots(1, 5, figsize=(12, 3))
    for i, (f, phase_diff) in enumerate(zip(f_list, phase_diff_list)):
        axs[i].plot(f, phase_diff)
        axs[i].set_title(f"Phase Diff: {tag_list[i]}")
        axs[i].set_xlabel("Frequency [Hz]")
        axs[i].set_ylabel("Phase Diff")

    plt.tight_layout()
    plt.show()


mapping = {
    "RF vs RM": "should be large",
    "RF vs RH": "should be small",
    "RF vs LF": "should be large",
    "RF vs LM": "should be small",
    "RF vs LH": "should be large",
    "RM vs RH": "should be large",
    "RM vs LF": "should be small",
    "RM vs LM": "should be large",
    "RM vs LH": "should be small",
    "RH vs LF": "should be large",
    "RH vs LM": "should be small",
    "RH vs LH": "should be large",
    "LF vs LM": "should be large",
    "LF vs LH": "should be small",
    "LM vs LH": "should be large",
}

if __name__ == "__main__":
    if leg_rf is not None:
        leg_rf.resample_phase_based_on_timestamp()
    if leg_lf is not None:
        leg_lf.resample_phase_based_on_timestamp()
    if leg_rm is not None:
        leg_rm.resample_phase_based_on_timestamp()
    if leg_lm is not None:
        leg_lm.resample_phase_based_on_timestamp()
    if leg_rh is not None:
        leg_rh.resample_phase_based_on_timestamp()
    if leg_lh is not None:
        leg_lh.resample_phase_based_on_timestamp()

    fs, common_time_grid = get_common_freq(
        leg_rf, leg_rm, leg_rh, leg_lf, leg_lm, leg_lh
    )

    if leg_rf is not None:
        leg_rf.resample_phase_based_on_common_time_grid(common_time_grid)
    if leg_lf is not None:
        leg_lf.resample_phase_based_on_common_time_grid(common_time_grid)
    if leg_rm is not None:
        leg_rm.resample_phase_based_on_common_time_grid(common_time_grid)
    if leg_lm is not None:
        leg_lm.resample_phase_based_on_common_time_grid(common_time_grid)
    if leg_rh is not None:
        leg_rh.resample_phase_based_on_common_time_grid(common_time_grid)
    if leg_lh is not None:
        leg_lh.resample_phase_based_on_common_time_grid(common_time_grid)

    leg_list = [leg_rf, leg_rm, leg_rh, leg_lf, leg_lm, leg_lh]
    leg_list = [leg for leg in leg_list if leg is not None]

    top_abs_freq_list = []  # List of top 3 dominant frequencies
    top_abs_freq_indices_list = []
    all_Cxy = []
    for i, each_leg in enumerate(leg_list):
        inner_f_list = []
        inner_Cxy_list = []
        tag_list = []
        for j, other_leg in enumerate(leg_list):
            if i < j:
                f, Cxy = calculate_magnitude_squared_coherence(
                    each_leg, other_leg, fs, nperseg=256
                )
                print(f"{each_leg.leg_name} vs {other_leg.leg_name}")
                inner_f_list.append(f)
                inner_Cxy_list.append(Cxy)
                tag_list.append(f"{each_leg.leg_name} vs {other_leg.leg_name}")

        print()
        if len(inner_f_list) != 0:
            # plot_msc(inner_f_list, inner_Cxy_list, tag_list)
            peaks_list = []
            for each in inner_Cxy_list:
                peaks_list.append(find_peaks(each)[0])

            print(len(peaks_list), len(inner_Cxy_list))
            top_freq_indices = []
            top_abs_freq_indices = []
            for i, each in enumerate(peaks_list):
                peak_values = inner_Cxy_list[i][each]
                top_in_peaks = np.argsort(peak_values)
                top_dominant_freq = np.argsort(inner_Cxy_list[i][each])[-7:]
                top_freq_indices.append(np.array(top_dominant_freq))

                top_abs_freq_indices.append(each[top_dominant_freq])

                # exit()
                # print(f"Top 3 Dominant Frequencies Index: {top_dominant_freq}")
                # print(f"Top 3 Freq's Coherence: {inner_Cxy_list[i][top_dominant_freq]}")
                # print(
                #     f"Top 3 Dominant Frequencies: {inner_f_list[i][top_dominant_freq]}"
                # )

            # plot_msc_with_peaks(
            #     inner_f_list, inner_Cxy_list, tag_list, peaks_list, top_freq_indices
            # )
            print()

            top_abs_freq_list.append(
                np.array([inner_f_list[i][x] for x in top_abs_freq_indices])
            )
            top_abs_freq_indices_list.append(top_abs_freq_indices)
            all_Cxy.append(inner_Cxy_list)

    print(top_abs_freq_list)
    print(top_abs_freq_indices_list)

    # save phase_diff for the later use
    tag_lists = []
    f_lists = []
    phase_diff_lists = []
    for i, each_leg in enumerate(leg_list):
        inner_f_list_2 = []
        inner_Diff_list = []
        tag_list = []
        for j, other_leg in enumerate(leg_list):
            if i < j:
                f, phase_diff = get_phase_diff_between_pair_of_legs(
                    each_leg, other_leg, fs, nperseg=256
                )
                # print(phase_diff)
                # print(f"{each_leg.leg_name} vs {other_leg.leg_name}")
                inner_f_list_2.append(f)
                inner_Diff_list.append(phase_diff)
                tag_list.append(f"{each_leg.leg_name} vs {other_leg.leg_name}")

        # print()
        if len(inner_f_list_2) != 0:
            # plot_phase_diff(inner_f_list_2, inner_Diff_list, tag_list)
            tag_lists.append(tag_list)
            f_lists.append(inner_f_list_2)
            phase_diff_lists.append(inner_Diff_list)
            # print()

    # how to get the phase diff based on freq
    print(top_abs_freq_list)
    print(top_abs_freq_indices_list)
    print(len(phase_diff_lists))
    print(len(phase_diff_lists[0]))
    print(len(phase_diff_lists[0][0]))
    print(len(f_lists))
    print(len(tag_lists))

    fig, axes = plt.subplots(len(tag_lists), len(tag_lists[0]), figsize=(12, 8))

    for i, each_index in enumerate(top_abs_freq_indices_list):
        # print(i, each_index)
        for j in range(len(each_index)):
            # print(j, each_index[j])
            most_top = each_index[j][0]
            # print(most_top)
            print(f"{tag_lists[i][j]}, {mapping[tag_lists[i][j]] }")
            for each_top in each_index[j]:
                print(
                    "Phase Diff: ",
                    abs(phase_diff_lists[i][j][each_top] / np.pi),
                    " π;",
                    "Frequency: ",
                    f_lists[i][j][each_top],
                    " Hz;",
                    "Coherence: ",
                    all_Cxy[i][j][each_top],
                )
                print()

            phase_diff_values = [
                abs(phase_diff_lists[i][j][each_top] / np.pi)
                for each_top in each_index[j]
            ]
            frequencies = [f_lists[i][j][each_top] for each_top in each_index[j]]
            coherences = [all_Cxy[i][j][each_top] for each_top in each_index[j]]

            scatter = axes[i, j].scatter(
                frequencies, coherences, c=phase_diff_values, cmap="viridis", s=10
            )
            axes[i, j].set_title(
                f"{tag_lists[i][j]} ({mapping[tag_lists[i][j]]})", size=8
            )
            axes[i, j].set_xlabel("Frequency (Hz)", size=4)
            axes[i, j].set_ylabel("Coherence", size=4)
            cbar = fig.colorbar(scatter, ax=axes[i, j])
            cbar.set_label("Phase Difference (π)", fontsize=4)
            cbar.ax.tick_params(labelsize=4)
            axes[i, j].tick_params(axis="y", labelsize=8)

    plt.tight_layout()
    plt.savefig(f"msc/{args.file}_msc.png")
    # plt.show()
