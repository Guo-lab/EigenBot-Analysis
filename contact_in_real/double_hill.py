import argparse
import itertools
import json
import test

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import entropy
from tqdm import tqdm


def scale_to_one_list(lst):
    min_val = min(lst)
    max_val = max(lst)
    scaled_list = [(x - min_val) / (max_val - min_val) for x in lst]
    return scaled_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--file",
        type=str,
        default="",
        help="The name of the file to read to calculate the entropy and histogram",
    )
    args = parser.parse_args()

    df_all = pd.read_csv(f"{args.file}_data.csv")
    print(f"=====================READ Data: \n{df_all}\n======================\n")

    df_all.sort_values(by="Timestamp", inplace=True)
    leg_data_rf = df_all[df_all["Leg"] == 13]["Phase"].to_numpy()
    leg_data_rm = df_all[df_all["Leg"] == 19]["Phase"].to_numpy()
    leg_data_rh = df_all[df_all["Leg"] == 17]["Phase"].to_numpy()
    leg_timestamp_rf = df_all[df_all["Leg"] == 13]["Timestamp"].to_numpy()
    leg_timestamp_rm = df_all[df_all["Leg"] == 19]["Timestamp"].to_numpy()
    leg_timestamp_rh = df_all[df_all["Leg"] == 17]["Timestamp"].to_numpy()

    print(f"Right Front Leg Data: \n{leg_data_rf} with length {len(leg_data_rf)}\n")
    print(f"Timestamp: \n{leg_timestamp_rf} with length {len(leg_timestamp_rf)}\n")

    print(f"Right Middle Leg Data: \n{leg_data_rm} with length {len(leg_data_rm)}\n")
    print(f"Timestamp: \n{leg_timestamp_rm} with length {len(leg_timestamp_rm)}\n")

    print(f"Right Hind Leg Data: \n{leg_data_rh} with length {len(leg_data_rh)}\n")
    print(f"Timestamp: \n{leg_timestamp_rh} with length {len(leg_timestamp_rh)}\n")

    # NOTE: Assume the Swing-Stance Phase is 8 seconds based on the box plot
    window_length_seconds = 8
    transitions = np.where((leg_data_rf[:-1] == 0) & (leg_data_rf[1:] == 1))[0] + 1
    if leg_data_rf[0] == 1:
        transitions = np.insert(transitions, 0, 0)
    print(f"Transitions: \n{transitions} with length {len(transitions)}\n")
    windows = []
    window_timestamps = []

    for i in transitions:
        start_time = leg_timestamp_rf[i]
        end_time = start_time + window_length_seconds
        window_front = leg_data_rf[
            (leg_timestamp_rf >= start_time) & (leg_timestamp_rf <= end_time)
        ]
        window_middle = leg_data_rm[
            (leg_timestamp_rm >= start_time) & (leg_timestamp_rm <= end_time)
        ]
        window_hind = leg_data_rh[
            (leg_timestamp_rh >= start_time) & (leg_timestamp_rh <= end_time)
        ]

        windows.append((window_front, window_middle, window_hind))
        window_timestamps.append(
            (
                leg_timestamp_rf[
                    (leg_timestamp_rf >= start_time) & (leg_timestamp_rf <= end_time)
                ],
                leg_timestamp_rm[
                    (leg_timestamp_rm >= start_time) & (leg_timestamp_rm <= end_time)
                ],
                leg_timestamp_rh[
                    (leg_timestamp_rh >= start_time) & (leg_timestamp_rh <= end_time)
                ],
            )
        )

    stacked_front = []
    stacked_middle = []
    stacked_hind = []
    stacked_time_front = []
    stacked_time_middle = []
    stacked_time_hind = []
    for window_front, window_middle, window_hind in windows:
        stacked_front.extend(window_front)
        stacked_middle.extend(window_middle)
        stacked_hind.extend(window_hind)

    for time_front, time_middle, time_hind in window_timestamps:
        time_front = time_front - time_front[0]
        time_middle = time_middle - time_middle[0]
        time_hind = time_hind - time_hind[0]
        stacked_time_front.extend(time_front)
        stacked_time_middle.extend(time_middle)
        stacked_time_hind.extend(time_hind)

    stacked_front = np.array(stacked_front)
    stacked_middle = np.array(stacked_middle)
    stacked_hind = np.array(stacked_hind)
    stacked_time_front = np.array(stacked_time_front)
    stacked_time_middle = np.array(stacked_time_middle)
    stacked_time_hind = np.array(stacked_time_hind)

    assert len(stacked_middle) == len(stacked_time_middle)
    assert len(stacked_front) == len(stacked_time_front)
    assert len(stacked_hind) == len(stacked_time_hind)

    time_bins = np.linspace(
        0, window_length_seconds, num=window_length_seconds * 10
    )  # 80 ms
    swing_counts_front = np.zeros_like(time_bins)
    swing_counts_middle = np.zeros_like(time_bins)
    swing_counts_hind = np.zeros_like(time_bins)

    for time, phase in zip(stacked_time_front, stacked_front):
        if phase == 0:
            bin_index = np.digitize(time, time_bins) - 1
            swing_counts_front[bin_index] += 1

    for time, phase in zip(stacked_time_middle, stacked_middle):
        if phase == 0:
            bin_index = np.digitize(time, time_bins) - 1
            swing_counts_middle[bin_index] += 1

    for time, phase in zip(stacked_time_hind, stacked_hind):
        if phase == 0:
            bin_index = np.digitize(time, time_bins) - 1
            swing_counts_hind[bin_index] += 1

    print(
        f"Swing Counts Front: \n{swing_counts_front} with length {len(swing_counts_front)}\n"
    )
    print(
        f"Swing Counts Middle: \n{swing_counts_middle} with length {len(swing_counts_middle)}\n"
    )
    print(
        f"Swing Counts Hind: \n{swing_counts_hind} with length {len(swing_counts_hind)}\n"
    )

    swing_counts_front = scale_to_one_list(swing_counts_front)
    swing_counts_middle = scale_to_one_list(swing_counts_middle)
    swing_counts_hind = scale_to_one_list(swing_counts_hind)

    plt.figure(figsize=(12, 5))
    bar_width = 0.06
    positions_front = time_bins - bar_width
    positions_hind = time_bins
    positions_middle = time_bins + bar_width
    plt.bar(
        positions_front,
        swing_counts_front,
        width=bar_width,
        color="blue",
        alpha=0.4,
        label="Right Front Leg",
    )
    plt.bar(
        positions_middle,
        swing_counts_middle,
        width=bar_width,
        color="red",
        alpha=0.4,
        label="Right Middle Leg",
    )
    plt.bar(
        positions_hind,
        swing_counts_hind,
        width=bar_width,
        color="purple",
        alpha=0.4,
        label="Right Hind Leg",
    )

    plt.title("Swing Phase Frequency in Right Legs Over Time")
    plt.xlabel("Window Over Time (s)")
    plt.ylabel("Swing Probability in the Stacked Window")

    plt.legend()

    plt.tight_layout()
    plt.savefig(f"{args.file}_swing_phase_combined.png")
    plt.show()
