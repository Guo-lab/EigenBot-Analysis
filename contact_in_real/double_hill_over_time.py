import argparse
import itertools
import json
import math
import test

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import seaborn as sns
from double_hill import scale_to_one_list
from scipy.stats import entropy
from tqdm import tqdm
from visualize import (create_animation, distance_divergence_plot,
                       distance_plot, divergence_plot, double_hill_plot)


def norm_scale(data):
    return data / np.sum(data) if np.sum(data) != 0 else data


def KL(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    return np.sum(np.where(a != 0, a * np.log(a / b), 0))


def KL_divergence(p, q):
    # print(p, q)
    p[p == 0], q[q == 0] = 1e-6, 1e-6
    vec = scipy.special.rel_entr(p, q)
    kl_div = np.sum(vec)
    # print(f"KL Divergence: {kl_div}")
    return kl_div


def calculate_probability_ipsi(
    transitions,
    window_length_seconds,
    leg_timestamp_rf,
    leg_timestamp_rm,
    leg_timestamp_rh,
    leg_data_rf,
    leg_data_rm,
    leg_data_rh,
    time_bins,
):
    # print(
    #     transitions,
    #     window_length_seconds,
    #     leg_timestamp_rf,
    #     leg_timestamp_rm,
    #     leg_timestamp_rh,
    #     leg_data_rf,
    #     leg_data_rm,
    #     leg_data_rh,
    # )
    inner_windows = []
    inner_window_timestamps = []

    for i in transitions:
        start_time = leg_timestamp_rf[i]
        end_time = start_time + window_length_seconds
        # print(f"Start Time: {start_time}, End Time: {end_time}")
        window_front = leg_data_rf[
            (leg_timestamp_rf >= start_time) & (leg_timestamp_rf <= end_time)
        ]
        window_middle = leg_data_rm[
            (leg_timestamp_rm >= start_time) & (leg_timestamp_rm <= end_time)
        ]
        window_hind = leg_data_rh[
            (leg_timestamp_rh >= start_time) & (leg_timestamp_rh <= end_time)
        ]
        inner_windows.append((window_front, window_middle, window_hind))
        inner_window_timestamps.append(
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

    for (window_front, window_middle, window_hind), (
        time_front,
        time_middle,
        time_hind,
    ) in zip(inner_windows, inner_window_timestamps):
        time_front -= time_front[0]
        if time_middle.size != 0:
            time_middle -= time_middle[0]
        if time_hind.size != 0:
            time_hind -= time_hind[0]

        if (
            len(window_front) == len(time_front)
            and len(window_middle) == len(time_middle)
            and len(window_hind) == len(time_hind)
        ):
            stacked_front.extend(window_front)
            stacked_middle.extend(window_middle)
            stacked_hind.extend(window_hind)
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

    # print(
    #     f"Swing Counts Front: {swing_counts_front}, with length {len(swing_counts_front)}"
    # )
    # print(
    #     f"Swing Counts Middle: {swing_counts_middle}, with length {len(swing_counts_middle)}"
    # )
    # print(
    #     f"Swing Counts Hind: {swing_counts_hind}, with length {len(swing_counts_hind)}"
    # )
    # swing_counts_front = scale_to_one_list(swing_counts_front)
    # swing_counts_middle = scale_to_one_list(swing_counts_middle)
    # swing_counts_hind = scale_to_one_list(swing_counts_hind)

    return (
        np.array(swing_counts_front),
        np.array(swing_counts_middle),
        np.array(swing_counts_hind),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--group",
        type=str,
        default="",
        help="The name of the file to read to calculate the entropy and histogram",
    )
    parser.add_argument(
        "--side",
        type=str,
        default="",
    )
    args = parser.parse_args()

    df_all = pd.read_csv(f"{args.group}_data.csv")
    df_all.sort_values(by="Timestamp", inplace=True)
    if args.side == "right":
        leg_data_rf = df_all[df_all["Leg"] == 13]["Phase"].to_numpy()
        leg_data_rm = df_all[df_all["Leg"] == 19]["Phase"].to_numpy()
        leg_data_rh = df_all[df_all["Leg"] == 17]["Phase"].to_numpy()
        leg_timestamp_rf = df_all[df_all["Leg"] == 13]["Timestamp"].to_numpy()
        leg_timestamp_rm = df_all[df_all["Leg"] == 19]["Timestamp"].to_numpy()
        leg_timestamp_rh = df_all[df_all["Leg"] == 17]["Timestamp"].to_numpy()
    elif args.side == "left":
        leg_data_rf = df_all[df_all["Leg"] == 5]["Phase"].to_numpy()
        leg_data_rm = df_all[df_all["Leg"] == 2]["Phase"].to_numpy()
        leg_data_rh = df_all[df_all["Leg"] == 18]["Phase"].to_numpy()
        leg_timestamp_rf = df_all[df_all["Leg"] == 5]["Timestamp"].to_numpy()
        leg_timestamp_rm = df_all[df_all["Leg"] == 2]["Timestamp"].to_numpy()
        leg_timestamp_rh = df_all[df_all["Leg"] == 18]["Timestamp"].to_numpy()

    # if len(leg_data_rf) == 0 or len(leg_data_rm) == 0 or len(leg_data_rh) == 0:
    #     print("No data for the specified side")
    #     exit(1)

    # NOTE: Assume the Swing-Stance Phase is 8 seconds based on the box plot
    window_length_seconds = 8
    transitions = np.where((leg_data_rf[:-1] == 0) & (leg_data_rf[1:] == 1))[0] + 1
    if leg_data_rf[0] == 1:
        transitions = np.insert(transitions, 0, 0)

    print(f"Transitions: \n{transitions} with length {len(transitions)}\n")
    min_timestamp = np.floor(leg_timestamp_rf.min() / 500) * 500
    max_timestamp = np.ceil(leg_timestamp_rf.max() / 500) * 500
    ranges = np.arange(min_timestamp, max_timestamp + 500, 500)
    print(f"Ranges: \n{ranges} with length {len(ranges)}\n")

    # Segment the data based on the range bin 500 (timestamp)
    segmented_data = []
    segmented_transitions = []
    for i in range(len(ranges) - 1):
        start_range = ranges[i]
        end_range = ranges[i + 1]
        indices_in_range = np.where(
            (leg_timestamp_rf >= start_range) & (leg_timestamp_rf < end_range)
        )[0]

        if len(indices_in_range) > 0:
            segmented_data.append(leg_data_rf[indices_in_range])
            segmented_transitions.append(
                transitions[
                    (transitions >= indices_in_range.min())
                    & (transitions <= indices_in_range.max())
                ]
            )

    # segmented_transitions:
    # [array([  0,  11,  38,  63,  87, 115, 141, 172, 198, 222, 248, 276, 307, 350]), array([362, 377, 405, 428, 452, 479, 507, 532, 561, 595, 615, 645, 672, 698, 724]), ... with length 5
    # leg_timestamp_rf[segmented_transitions[0]]:
    # Timestamps: [504.268 507.341 514.254 520.655 526.799 533.969 540.625 548.565 555.224 561.624 568.285 575.71  583.904 594.914] with length 14

    slide_window = []
    slide_window_length_seconds = 40
    total_time = int(np.max(leg_timestamp_rf % 500))

    gif_front_swing_prob = []
    gif_middle_swing_prob = []
    gif_hind_swing_prob = []
    time_bins = np.linspace(0, window_length_seconds, num=window_length_seconds * 10)

    from scipy.stats import wasserstein_distance

    KL_divergence_middle_from_front = []
    KL_divergence_hind_from_front = []
    wasserstein_distance_front_middle = []
    wasserstein_distance_front_hind = []
    wasserstein_distance_middle_hind = []

    for start_t in range(0, total_time):
        end_t = start_t + slide_window_length_seconds

        swing_front_cnt_array = np.zeros(window_length_seconds * 10)
        swing_middle_cnt_array = np.zeros(window_length_seconds * 10)
        swing_hind_cnt_array = np.zeros(window_length_seconds * 10)
        for each_transition in segmented_transitions:
            timestamp_in_segment = leg_timestamp_rf[each_transition] % 500  # offset 500
            # print(f"Timestamps: {timestamp_in_segment}")
            # print(f"Transition: {each_transition}")
            indices_in_range = np.where(
                (timestamp_in_segment >= start_t) & (timestamp_in_segment <= end_t)
            )[0]
            # print(f"Indices in Range: {indices_in_range}")
            curr_transition = each_transition[indices_in_range]
            # print(f"Current Transition: {curr_transition}")

            swing_counts_front, swing_counts_middle, swing_counts_hind = (
                calculate_probability_ipsi(
                    curr_transition,
                    window_length_seconds,
                    leg_timestamp_rf,
                    leg_timestamp_rm,
                    leg_timestamp_rh,
                    leg_data_rf,
                    leg_data_rm,
                    leg_data_rh,
                    time_bins,
                )
            )
            if len(swing_front_cnt_array) == 0:
                swing_front_cnt_array = swing_counts_front
                swing_middle_cnt_array = swing_counts_middle
                swing_hind_cnt_array = swing_counts_hind
            else:
                swing_front_cnt_array = swing_front_cnt_array + swing_counts_front
                swing_middle_cnt_array = swing_middle_cnt_array + swing_counts_middle
                swing_hind_cnt_array = swing_hind_cnt_array + swing_counts_hind

        if np.sum(swing_front_cnt_array) != 0:
            # swing_front_cnt_array = norm_scale(swing_front_cnt_array)
            # swing_middle_cnt_array = norm_scale(swing_middle_cnt_array)
            # swing_hind_cnt_array = norm_scale(swing_hind_cnt_array)
            swing_front_cnt_array = np.array(scale_to_one_list(swing_front_cnt_array))
            gif_front_swing_prob.append(swing_front_cnt_array)

        if np.sum(swing_middle_cnt_array) != 0:
            swing_middle_cnt_array = np.array(scale_to_one_list(swing_middle_cnt_array))
            gif_middle_swing_prob.append(swing_middle_cnt_array)

        if np.sum(swing_hind_cnt_array) != 0:
            swing_hind_cnt_array = np.array(scale_to_one_list(swing_hind_cnt_array))
            gif_hind_swing_prob.append(swing_hind_cnt_array)

        if np.sum(swing_front_cnt_array) != 0 and np.sum(swing_middle_cnt_array) != 0:
            KL_divergence_middle_from_front.append(
                KL_divergence(
                    norm_scale(swing_middle_cnt_array),
                    norm_scale(swing_front_cnt_array),
                )
            )
            wasserstein_distance_front_middle.append(
                wasserstein_distance(
                    norm_scale(swing_middle_cnt_array),
                    norm_scale(swing_front_cnt_array),
                )
            )
        else:
            KL_divergence_middle_from_front.append(None)
            wasserstein_distance_front_middle.append(None)

        if np.sum(swing_front_cnt_array) != 0 and np.sum(swing_hind_cnt_array) != 0:
            KL_divergence_hind_from_front.append(
                KL_divergence(
                    norm_scale(swing_hind_cnt_array),
                    norm_scale(swing_front_cnt_array),
                )
            )
            wasserstein_distance_front_hind.append(
                wasserstein_distance(
                    norm_scale(swing_hind_cnt_array),
                    norm_scale(swing_front_cnt_array),
                )
            )
        else:
            KL_divergence_hind_from_front.append(None)
            wasserstein_distance_front_hind.append(None)

        if np.sum(swing_middle_cnt_array) != 0 and np.sum(swing_hind_cnt_array) != 0:
            wasserstein_distance_middle_hind.append(
                wasserstein_distance(
                    norm_scale(swing_middle_cnt_array),
                    norm_scale(swing_hind_cnt_array),
                )
            )
        else:
            wasserstein_distance_middle_hind.append(None)

    # print(
    #     f"Front Swing Prob: {gif_front_swing_prob}, with length {len(gif_front_swing_prob)}"
    # )
    # print(
    #     f"Middle Swing Prob: {gif_middle_swing_prob}, with length {len(gif_middle_swing_prob)}"
    # )
    # print(
    #     f"Hind Swing Prob: {gif_hind_swing_prob}, with length {len(gif_hind_swing_prob)}"
    # )

    create_animation(
        args,
        time_bins,
        gif_front_swing_prob,
        gif_middle_swing_prob,
        gif_hind_swing_prob,
    )

    distance_divergence_plot(
        args,
        KL_divergence_middle_from_front,
        KL_divergence_hind_from_front,
        wasserstein_distance_front_middle,
        wasserstein_distance_front_hind,
        wasserstein_distance_middle_hind,
    )

    divergence_plot(
        args, KL_divergence_middle_from_front, KL_divergence_hind_from_front
    )
    distance_plot(
        args,
        wasserstein_distance_front_middle,
        wasserstein_distance_front_hind,
        wasserstein_distance_middle_hind,
    )
