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
from visualize import all_legs_hills


def scale_to_one_list(lst):
    min_val = min(lst)
    max_val = max(lst)
    scaled_list = [(x - min_val) / (max_val - min_val) for x in lst]
    return scaled_list


def norm_scale(data):
    return data / np.sum(data) if np.sum(data) != 0 else data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--group",
        type=str,
        default="",
        help="The name of the file to read to calculate the entropy and histogram",
    )
    args = parser.parse_args()
    df_all = pd.read_csv(f"{args.group}_data.csv")
    df_all.sort_values(by="Timestamp", inplace=True)
    leg_data_rf = df_all[df_all["Leg"] == 13]["Phase"].to_numpy()
    leg_data_rm = df_all[df_all["Leg"] == 19]["Phase"].to_numpy()
    leg_data_rh = df_all[df_all["Leg"] == 17]["Phase"].to_numpy()
    leg_timestamp_rf = df_all[df_all["Leg"] == 13]["Timestamp"].to_numpy()
    leg_timestamp_rm = df_all[df_all["Leg"] == 19]["Timestamp"].to_numpy()
    leg_timestamp_rh = df_all[df_all["Leg"] == 17]["Timestamp"].to_numpy()
    leg_data_lf = df_all[df_all["Leg"] == 5]["Phase"].to_numpy()
    leg_data_lm = df_all[df_all["Leg"] == 2]["Phase"].to_numpy()
    leg_data_lh = df_all[df_all["Leg"] == 18]["Phase"].to_numpy()
    leg_timestamp_lf = df_all[df_all["Leg"] == 5]["Timestamp"].to_numpy()
    leg_timestamp_lm = df_all[df_all["Leg"] == 2]["Timestamp"].to_numpy()
    leg_timestamp_lh = df_all[df_all["Leg"] == 18]["Timestamp"].to_numpy()

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
        window_front_2 = leg_data_lf[
            (leg_timestamp_lf >= start_time) & (leg_timestamp_lf <= end_time)
        ]
        window_middle_2 = leg_data_lm[
            (leg_timestamp_lm >= start_time) & (leg_timestamp_lm <= end_time)
        ]
        window_hind_2 = leg_data_lh[
            (leg_timestamp_lh >= start_time) & (leg_timestamp_lh <= end_time)
        ]

        windows.append(
            (
                window_front,
                window_middle,
                window_hind,
                window_front_2,
                window_middle_2,
                window_hind_2,
            )
        )
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
                leg_timestamp_lf[
                    (leg_timestamp_lf >= start_time) & (leg_timestamp_lf <= end_time)
                ],
                leg_timestamp_lm[
                    (leg_timestamp_lm >= start_time) & (leg_timestamp_lm <= end_time)
                ],
                leg_timestamp_lh[
                    (leg_timestamp_lh >= start_time) & (leg_timestamp_lh <= end_time)
                ],
            )
        )

    stacked_front = []
    stacked_middle = []
    stacked_hind = []
    stacked_front_2 = []
    stacked_middle_2 = []
    stacked_hind_2 = []
    stacked_time_front = []
    stacked_time_middle = []
    stacked_time_hind = []
    stacked_time_front_2 = []
    stacked_time_middle_2 = []
    stacked_time_hind_2 = []
    for (
        window_front,
        window_middle,
        window_hind,
        window_front_2,
        window_middle_2,
        window_hind_2,
    ) in windows:
        stacked_front.extend(window_front)
        stacked_middle.extend(window_middle)
        stacked_hind.extend(window_hind)
        stacked_front_2.extend(window_front_2)
        stacked_middle_2.extend(window_middle_2)
        stacked_hind_2.extend(window_hind_2)

    for (
        time_front,
        time_middle,
        time_hind,
        time_front_2,
        time_middle_2,
        time_hind_2,
    ) in window_timestamps:
        time_front = time_front - time_front[0]
        if len(time_middle) != 0:
            time_middle = time_middle - time_middle[0]
        if len(time_hind) != 0:
            time_hind = time_hind - time_hind[0]
        time_front_2 = time_front_2 - time_front_2[0]
        if len(time_middle_2) != 0:
            time_middle_2 = time_middle_2 - time_middle_2[0]
        if len(time_hind_2) != 0:
            time_hind_2 = time_hind_2 - time_hind_2[0]
        stacked_time_front.extend(time_front)
        stacked_time_middle.extend(time_middle)
        stacked_time_hind.extend(time_hind)
        stacked_time_front_2.extend(time_front_2)
        stacked_time_middle_2.extend(time_middle_2)
        stacked_time_hind_2.extend(time_hind_2)

    stacked_front = np.array(stacked_front)
    stacked_middle = np.array(stacked_middle)
    stacked_hind = np.array(stacked_hind)
    stacked_time_front = np.array(stacked_time_front)
    stacked_time_middle = np.array(stacked_time_middle)
    stacked_time_hind = np.array(stacked_time_hind)
    stacked_time_front_2 = np.array(stacked_time_front_2)
    stacked_time_middle_2 = np.array(stacked_time_middle_2)
    stacked_time_hind_2 = np.array(stacked_time_hind_2)

    assert len(stacked_middle) == len(stacked_time_middle)
    assert len(stacked_front) == len(stacked_time_front)
    assert len(stacked_hind) == len(stacked_time_hind)

    time_bins = np.linspace(
        0, window_length_seconds, num=window_length_seconds * 10
    )  # 80 ms
    swing_counts_front = np.zeros_like(time_bins)
    swing_counts_middle = np.zeros_like(time_bins)
    swing_counts_hind = np.zeros_like(time_bins)
    swing_counts_front_2 = np.zeros_like(time_bins)
    swing_counts_middle_2 = np.zeros_like(time_bins)
    swing_counts_hind_2 = np.zeros_like(time_bins)

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

    for time, phase in zip(stacked_time_front_2, stacked_front_2):
        if phase == 0:
            bin_index = np.digitize(time, time_bins) - 1
            swing_counts_front_2[bin_index] += 1

    for time, phase in zip(stacked_time_middle_2, stacked_middle_2):
        if phase == 0:
            bin_index = np.digitize(time, time_bins) - 1
            swing_counts_middle_2[bin_index] += 1

    for time, phase in zip(stacked_time_hind_2, stacked_hind_2):
        if phase == 0:
            bin_index = np.digitize(time, time_bins) - 1
            swing_counts_hind_2[bin_index] += 1

    # print(
    #     f"Swing Counts Front: \n{swing_counts_front} with length {len(swing_counts_front)}\n"
    # )
    # print(
    #     f"Swing Counts Middle: \n{swing_counts_middle} with length {len(swing_counts_middle)}\n"
    # )
    # print(
    #     f"Swing Counts Hind: \n{swing_counts_hind} with length {len(swing_counts_hind)}\n"
    # )

    # swing_counts_front = norm_scale(swing_counts_front)
    # swing_counts_middle = norm_scale(swing_counts_middle)
    # swing_counts_hind = norm_scale(swing_counts_hind)
    # swing_counts_front_2 = norm_scale(swing_counts_front_2)
    # swing_counts_middle_2 = norm_scale(swing_counts_middle_2)
    # swing_counts_hind_2 = norm_scale(swing_counts_hind_2)
    # swing_counts_front = scale_to_one_list(swing_counts_front)
    # swing_counts_middle = scale_to_one_list(swing_counts_middle)
    # swing_counts_hind = scale_to_one_list(swing_counts_hind)
    # swing_counts_front_2 = scale_to_one_list(swing_counts_front_2)
    # swing_counts_middle_2 = scale_to_one_list(swing_counts_middle_2)
    # swing_counts_hind_2 = scale_to_one_list(swing_counts_hind_2)
    print(swing_counts_front)
    print(len(swing_counts_front))
    print(len(windows))
    swing_counts_front = swing_counts_front / len(windows)
    swing_counts_middle = swing_counts_middle / len(windows)
    swing_counts_hind = swing_counts_hind / len(windows)
    swing_counts_front_2 = swing_counts_front_2 / len(windows)
    swing_counts_middle_2 = swing_counts_middle_2 / len(windows)
    swing_counts_hind_2 = swing_counts_hind_2 / len(windows)
    all_legs_hills(
        args,
        time_bins,
        swing_counts_front,
        swing_counts_middle,
        swing_counts_hind,
        swing_counts_front_2,
        swing_counts_middle_2,
        swing_counts_hind_2,
    )
