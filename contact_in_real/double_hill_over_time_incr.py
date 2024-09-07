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
from scipy.stats import entropy, wasserstein_distance
from tqdm import tqdm
from visualize import (create_animation, create_animation_incr,
                       distance_divergence_plot, distance_incr_plot,
                       distance_plot, divergence_incr_plot, divergence_plot,
                       double_hill_plot)


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


def pipeline_in_calculate_probability(
    time_window, window, time_bins, swing_counts_stack
):
    swing_counts = np.zeros_like(time_bins)
    for time, phase in zip(time_window, window):
        if phase == 0:
            bin_index = np.digitize(time, time_bins) - 1
            swing_counts[bin_index] += 1
    swing_counts[swing_counts != 0] = 1
    swing_counts_stack = np.vstack((swing_counts_stack, swing_counts))
    return swing_counts_stack


def calculate_probability_ipsi(
    transitions,
    window_length_seconds,
    leg_timestamp_lf,
    leg_timestamp_lm,
    leg_timestamp_lh,
    leg_data_lf,
    leg_data_lm,
    leg_data_lh,
    leg_timestamp_rf,
    leg_timestamp_rm,
    leg_timestamp_rh,
    leg_data_rf,
    leg_data_rm,
    leg_data_rh,
    time_bins,
):
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
        window_front_2 = leg_data_lf[
            (leg_timestamp_lf >= start_time) & (leg_timestamp_lf <= end_time)
        ]
        window_middle_2 = leg_data_lm[
            (leg_timestamp_lm >= start_time) & (leg_timestamp_lm <= end_time)
        ]
        window_hind_2 = leg_data_lh[
            (leg_timestamp_lh >= start_time) & (leg_timestamp_lh <= end_time)
        ]
        inner_windows.append(
            (
                window_front,
                window_middle,
                window_hind,
                window_front_2,
                window_middle_2,
                window_hind_2,
            )
        )
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

    swing_counts_front_stack = np.zeros_like(time_bins)
    swing_counts_middle_stack = np.zeros_like(time_bins)
    swing_counts_hind_stack = np.zeros_like(time_bins)
    swing_counts_front_2_stack = np.zeros_like(time_bins)
    swing_counts_middle_2_stack = np.zeros_like(time_bins)
    swing_counts_hind_2_stack = np.zeros_like(time_bins)
    for (
        window_front,
        window_middle,
        window_hind,
        window_front_2,
        window_middle_2,
        window_hind_2,
    ), (
        time_front,
        time_middle,
        time_hind,
        time_front_2,
        time_middle_2,
        time_hind_2,
    ) in zip(
        inner_windows, inner_window_timestamps
    ):
        time_front -= time_front[0]
        if len(time_middle) != 0:
            time_middle -= time_middle[0]
        if len(time_hind) != 0:
            time_hind -= time_hind[0]
        time_front_2 -= time_front_2[0]
        if len(time_middle_2) != 0:
            time_middle_2 -= time_middle_2[0]
        if len(time_hind_2) != 0:
            time_hind_2 -= time_hind_2[0]

        swing_counts_front_stack = pipeline_in_calculate_probability(
            time_front, window_front, time_bins, swing_counts_front_stack
        )
        swing_counts_middle_stack = pipeline_in_calculate_probability(
            time_middle, window_middle, time_bins, swing_counts_middle_stack
        )
        swing_counts_hind_stack = pipeline_in_calculate_probability(
            time_hind, window_hind, time_bins, swing_counts_hind_stack
        )
        swing_counts_front_2_stack = pipeline_in_calculate_probability(
            time_front_2, window_front_2, time_bins, swing_counts_front_2_stack
        )
        swing_counts_middle_2_stack = pipeline_in_calculate_probability(
            time_middle_2, window_middle_2, time_bins, swing_counts_middle_2_stack
        )
        swing_counts_hind_2_stack = pipeline_in_calculate_probability(
            time_hind_2, window_hind_2, time_bins, swing_counts_hind_2_stack
        )

    return (
        np.mean(swing_counts_front_stack, axis=0),
        np.mean(swing_counts_middle_stack, axis=0),
        np.mean(swing_counts_hind_stack, axis=0),
        np.mean(swing_counts_front_2_stack, axis=0),
        np.mean(swing_counts_middle_2_stack, axis=0),
        np.mean(swing_counts_hind_2_stack, axis=0),
    )


def get_distance_divergence_incr(array_a, array_b, KL_list, wasserstein_list):
    if np.sum(array_a) != 0 and np.sum(array_b) != 0:
        KL_list.append(
            KL_divergence(
                norm_scale(array_b),
                norm_scale(array_a),
            )
        )
        wasserstein_list.append(
            wasserstein_distance(
                norm_scale(array_b),
                norm_scale(array_a),
            )
        )
    else:
        KL_list.append(None)
        wasserstein_list.append(None)

    return KL_list, wasserstein_list


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
    gif_front_swing_prob_2 = []
    gif_middle_swing_prob_2 = []
    gif_hind_swing_prob_2 = []
    bins_seconds_inverse = 8
    time_bins = np.linspace(
        0, window_length_seconds, num=window_length_seconds * bins_seconds_inverse
    )

    KL_divergence_middle_from_front = []
    KL_divergence_hind_from_front = []
    KL_divergence_front_2_from_front = []
    KL_divergence_middle_2_from_front = []
    KL_divergence_hind_2_from_front = []

    wasserstein_distance_front_middle = []
    wasserstein_distance_front_hind = []
    wasserstein_distance_front_front_2 = []
    wasserstein_distance_front_middle_2 = []
    wasserstein_distance_front_hind_2 = []

    incr = 0
    stat_swing_front = np.zeros(window_length_seconds * bins_seconds_inverse)
    stat_swing_middle = np.zeros(window_length_seconds * bins_seconds_inverse)
    stat_swing_hind = np.zeros(window_length_seconds * bins_seconds_inverse)
    stat_swing_front_2 = np.zeros(window_length_seconds * bins_seconds_inverse)
    stat_swing_middle_2 = np.zeros(window_length_seconds * bins_seconds_inverse)
    stat_swing_hind_2 = np.zeros(window_length_seconds * bins_seconds_inverse)
    for start_t in range(0, total_time):
        incr += 1
        end_t = start_t + slide_window_length_seconds

        swing_front_cnt_array = np.zeros(window_length_seconds * bins_seconds_inverse)
        swing_middle_cnt_array = np.zeros(window_length_seconds * bins_seconds_inverse)
        swing_hind_cnt_array = np.zeros(window_length_seconds * bins_seconds_inverse)
        swing_front_cnt_array_2 = np.zeros(window_length_seconds * bins_seconds_inverse)
        swing_middle_cnt_array_2 = np.zeros(
            window_length_seconds * bins_seconds_inverse
        )
        swing_hind_cnt_array_2 = np.zeros(window_length_seconds * bins_seconds_inverse)

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

            (
                swing_counts_front,
                swing_counts_middle,
                swing_counts_hind,
                swing_counts_front_2,
                swing_counts_middle_2,
                swing_counts_hind_2,
            ) = calculate_probability_ipsi(
                curr_transition,
                window_length_seconds,
                leg_timestamp_lf,
                leg_timestamp_lm,
                leg_timestamp_lh,
                leg_data_lf,
                leg_data_lm,
                leg_data_lh,
                leg_timestamp_rf,
                leg_timestamp_rm,
                leg_timestamp_rh,
                leg_data_rf,
                leg_data_rm,
                leg_data_rh,
                time_bins,
            )
            swing_front_cnt_array = swing_front_cnt_array + swing_counts_front
            swing_middle_cnt_array = swing_middle_cnt_array + swing_counts_middle
            swing_hind_cnt_array = swing_hind_cnt_array + swing_counts_hind
            swing_front_cnt_array_2 = swing_front_cnt_array_2 + swing_counts_front_2
            swing_middle_cnt_array_2 = swing_middle_cnt_array_2 + swing_counts_middle_2
            swing_hind_cnt_array_2 = swing_hind_cnt_array_2 + swing_counts_hind_2

        swing_front_cnt_array = swing_front_cnt_array / len(segmented_transitions)
        swing_middle_cnt_array = swing_middle_cnt_array / len(segmented_transitions)
        swing_hind_cnt_array = swing_hind_cnt_array / len(segmented_transitions)
        swing_front_cnt_array_2 = swing_front_cnt_array_2 / len(segmented_transitions)
        swing_middle_cnt_array_2 = swing_middle_cnt_array_2 / len(segmented_transitions)
        swing_hind_cnt_array_2 = swing_hind_cnt_array_2 / len(segmented_transitions)

        stat_swing_front = (
            stat_swing_front * (incr - 1) + swing_front_cnt_array
        ) / incr
        stat_swing_middle = (
            stat_swing_middle * (incr - 1) + swing_middle_cnt_array
        ) / incr
        stat_swing_hind = (stat_swing_hind * (incr - 1) + swing_hind_cnt_array) / incr
        stat_swing_front_2 = (
            stat_swing_front_2 * (incr - 1) + swing_front_cnt_array_2
        ) / incr
        stat_swing_middle_2 = (
            stat_swing_middle_2 * (incr - 1) + swing_middle_cnt_array_2
        ) / incr
        stat_swing_hind_2 = (
            stat_swing_hind_2 * (incr - 1) + swing_hind_cnt_array_2
        ) / incr

        if np.sum(stat_swing_front) != 0:
            gif_front_swing_prob.append(stat_swing_front)

        if np.sum(stat_swing_middle) != 0:
            gif_middle_swing_prob.append(stat_swing_middle)

        if np.sum(stat_swing_hind) != 0:
            gif_hind_swing_prob.append(stat_swing_hind)

        if np.sum(stat_swing_front_2) != 0:
            gif_front_swing_prob_2.append(stat_swing_front_2)

        if np.sum(stat_swing_middle_2) != 0:
            gif_middle_swing_prob_2.append(stat_swing_middle_2)

        if np.sum(stat_swing_hind_2) != 0:
            gif_hind_swing_prob_2.append(stat_swing_hind_2)

        KL_divergence_middle_from_front, wasserstein_distance_front_middle = (
            get_distance_divergence_incr(
                stat_swing_front,
                stat_swing_middle,
                KL_divergence_middle_from_front,
                wasserstein_distance_front_middle,
            )
        )

        KL_divergence_hind_from_front, wasserstein_distance_front_hind = (
            get_distance_divergence_incr(
                stat_swing_front,
                stat_swing_hind,
                KL_divergence_hind_from_front,
                wasserstein_distance_front_hind,
            )
        )

        KL_divergence_front_2_from_front, wasserstein_distance_front_front_2 = (
            get_distance_divergence_incr(
                stat_swing_front,
                stat_swing_front_2,
                KL_divergence_front_2_from_front,
                wasserstein_distance_front_front_2,
            )
        )

        KL_divergence_middle_2_from_front, wasserstein_distance_front_middle_2 = (
            get_distance_divergence_incr(
                stat_swing_front,
                stat_swing_middle_2,
                KL_divergence_middle_2_from_front,
                wasserstein_distance_front_middle_2,
            )
        )

        KL_divergence_hind_2_from_front, wasserstein_distance_front_hind_2 = (
            get_distance_divergence_incr(
                stat_swing_front,
                stat_swing_hind_2,
                KL_divergence_hind_2_from_front,
                wasserstein_distance_front_hind_2,
            )
        )

    # print(
    #     f"Front Swing Prob: {gif_front_swing_prob}, with length {len(gif_front_swing_prob)}"
    # )
    # print(
    #     f"Middle Swing Prob: {gif_middle_swing_prob}, with length {len(gif_middle_swing_prob)}"
    # )
    # print(
    #     f"Hind Swing Prob: {gif_hind_swing_prob}, with length {len(gif_hind_swing_prob)}"
    # )

    ########### NAME_LIST_8 Comment Below ###########
    create_animation_incr(
        args,
        time_bins,
        gif_front_swing_prob,
        gif_middle_swing_prob,
        gif_hind_swing_prob,
        gif_front_swing_prob_2,
        gif_middle_swing_prob_2,
        gif_hind_swing_prob_2,
    )

    print(
        len(gif_front_swing_prob),
        len(gif_middle_swing_prob),
        len(gif_hind_swing_prob),
        len(gif_front_swing_prob_2),
        len(gif_middle_swing_prob_2),
        len(gif_hind_swing_prob_2),
        gif_front_swing_prob[0].shape,
    )

    np.save("gif_front_swing_prob.npy", gif_front_swing_prob)
    np.save("gif_middle_swing_prob.npy", gif_middle_swing_prob)
    np.save("gif_hind_swing_prob.npy", gif_hind_swing_prob)
    np.save("gif_front_swing_prob_2.npy", gif_front_swing_prob_2)
    np.save("gif_middle_swing_prob_2.npy", gif_middle_swing_prob_2)
    np.save("gif_hind_swing_prob_2.npy", gif_hind_swing_prob_2)

    """
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

    """

    # divergence_incr_plot(
    #     args,
    #     KL_divergence_middle_from_front,
    #     KL_divergence_hind_from_front,
    #     KL_divergence_front_2_from_front,
    #     KL_divergence_middle_2_from_front,
    #     KL_divergence_hind_2_from_front,
    # )

    # distance_incr_plot(
    #     args,
    #     wasserstein_distance_front_middle,
    #     wasserstein_distance_front_hind,
    #     wasserstein_distance_front_front_2,
    #     wasserstein_distance_front_middle_2,
    #     wasserstein_distance_front_hind_2,
    # )
