import argparse
import itertools
import json
import test

import matplotlib.pyplot as plt
import matplotlib.scale as mscale
import numpy as np
import pandas as pd
import scipy
import seaborn as sns
from scipy.stats import entropy, wasserstein_distance
from tqdm import tqdm
from visualize import all_legs_hills


def scale_to_one_list(lst):
    min_val = min(lst)
    max_val = max(lst)
    scaled_list = [(x - min_val) / (max_val - min_val) for x in lst]
    return scaled_list


def norm_scale(data):
    return data / np.sum(data) if np.sum(data) != 0 else data


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


def KL_divergence(p, q):
    # print(p, q)
    p[p == 0], q[q == 0] = 1e-6, 1e-6
    vec = scipy.special.rel_entr(p, q)
    kl_div = np.sum(vec)
    # print(f"KL Divergence: {kl_div}")
    return kl_div


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


def get_KL_for_each_trial(each_trial):
    df_all = pd.read_csv(f"{each_trial}_data.csv")
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

    time_bins = np.linspace(0, window_length_seconds, num=window_length_seconds * 8)
    (
        swing_counts_front,
        swing_counts_middle,
        swing_counts_hind,
        swing_counts_front_2,
        swing_counts_middle_2,
        swing_counts_hind_2,
    ) = calculate_probability_ipsi(
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
    )

    # all_legs_hills(
    #     args,
    #     time_bins,
    #     swing_counts_front,
    #     swing_counts_middle,
    #     swing_counts_hind,
    #     swing_counts_front_2,
    #     swing_counts_middle_2,
    #     swing_counts_hind_2,
    # )

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

    print(swing_counts_front)
    print(swing_counts_middle)
    print(swing_counts_hind)
    print(swing_counts_front_2)
    print(swing_counts_middle_2)
    print(swing_counts_hind_2)

    KL_divergence_middle_from_front, wasserstein_distance_front_middle = (
        get_distance_divergence_incr(
            swing_counts_front,
            swing_counts_middle,
            KL_divergence_middle_from_front,
            wasserstein_distance_front_middle,
        )
    )

    KL_divergence_hind_from_front, wasserstein_distance_front_hind = (
        get_distance_divergence_incr(
            swing_counts_front,
            swing_counts_hind,
            KL_divergence_hind_from_front,
            wasserstein_distance_front_hind,
        )
    )

    KL_divergence_front_2_from_front, wasserstein_distance_front_front_2 = (
        get_distance_divergence_incr(
            swing_counts_front,
            swing_counts_front_2,
            KL_divergence_front_2_from_front,
            wasserstein_distance_front_front_2,
        )
    )

    KL_divergence_middle_2_from_front, wasserstein_distance_front_middle_2 = (
        get_distance_divergence_incr(
            swing_counts_front,
            swing_counts_middle_2,
            KL_divergence_middle_2_from_front,
            wasserstein_distance_front_middle_2,
        )
    )

    KL_divergence_hind_2_from_front, wasserstein_distance_front_hind_2 = (
        get_distance_divergence_incr(
            swing_counts_front,
            swing_counts_hind_2,
            KL_divergence_hind_2_from_front,
            wasserstein_distance_front_hind_2,
        )
    )

    print(f"KL Divergence Middle from Front: {KL_divergence_middle_from_front}")
    print(f"KL Divergence Hind from Front: {KL_divergence_hind_from_front}")
    print(f"KL Divergence Front 2 from Front: {KL_divergence_front_2_from_front}")
    print(f"KL Divergence Middle 2 from Front: {KL_divergence_middle_2_from_front}")
    print(f"KL Divergence Hind 2 from Front: {KL_divergence_hind_2_from_front}")
    return (
        KL_divergence_middle_from_front,
        KL_divergence_hind_from_front,
        KL_divergence_front_2_from_front,
        KL_divergence_middle_2_from_front,
        KL_divergence_hind_2_from_front,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--group",
        type=str,
        nargs="+",  # This allows the argument to accept multiple values as a list
        help="The names of the files to read to calculate the entropy and histogram",
    )
    parser.add_argument(
        "--tag",
        type=str,
        help="The tag to append to the file name",
    )

    args = parser.parse_args()
    print(args.group)

    KL_divergence_middle_from_front_list = []
    KL_divergence_hind_from_front_list = []
    KL_divergence_front_2_from_front_list = []
    KL_divergence_middle_2_from_front_list = []
    KL_divergence_hind_2_from_front_list = []

    for each_trial in args.group:
        x, y, z, m, n = get_KL_for_each_trial(each_trial)

        KL_divergence_middle_from_front_list.append(x)
        KL_divergence_hind_from_front_list.append(y)
        KL_divergence_front_2_from_front_list.append(z)
        KL_divergence_middle_2_from_front_list.append(m)
        KL_divergence_hind_2_from_front_list.append(n)

    x_axis = range(len(KL_divergence_middle_from_front_list))

    # Plotting
    plt.figure(figsize=(10, 6))  # Adjust the figure size if needed

    plt.scatter(
        x_axis,
        KL_divergence_middle_from_front_list,
        label="R2 from R1",
        color="blue",
    )
    plt.scatter(
        x_axis,
        KL_divergence_hind_from_front_list,
        label="R3 from R1",
        color="green",
    )
    plt.scatter(
        x_axis,
        KL_divergence_front_2_from_front_list,
        label="L1 from R1",
        color="red",
    )
    plt.scatter(
        x_axis,
        KL_divergence_middle_2_from_front_list,
        label="L2 from R1",
        color="orange",
    )
    plt.scatter(
        x_axis,
        KL_divergence_hind_2_from_front_list,
        label="L3 from R1",
        color="purple",
    )

    # plt.yscale(mscale.LogScale(axis="y", base=10))

    # Adding labels and title
    plt.xlabel("Trial Index")
    plt.ylabel("KL Divergence Value")
    plt.title(f"{args.tag} KL Divergence Across Trials")
    plt.legend(loc="best")

    # Show the plot
    plt.savefig(f"radius/{args.tag}_KL_Divergence_radius.png")
    # plt.show()

    # The mean of the data
    filtered_data = [x for x in KL_divergence_middle_from_front_list if x is not None]
    if filtered_data[0] != None:
        mean_KL_divergence_middle_from_front = (
            np.mean(filtered_data) if filtered_data else None
        )
    else:
        mean_KL_divergence_middle_from_front = None

    filtered_data = [x for x in KL_divergence_hind_from_front_list if x is not None]
    if filtered_data[0] != None:
        mean_KL_divergence_hind_from_front = (
            np.mean(filtered_data) if filtered_data else None
        )
    else:
        mean_KL_divergence_hind_from_front = None

    filtered_data = [x for x in KL_divergence_front_2_from_front_list if x is not None]
    if filtered_data[0] != None:
        mean_KL_divergence_front_2_from_front = (
            np.mean(filtered_data) if filtered_data else None
        )
    else:
        mean_KL_divergence_front_2_from_front = None

    filtered_data = [x for x in KL_divergence_middle_2_from_front_list if x is not None]
    if filtered_data[0] != None:
        mean_KL_divergence_middle_2_from_front = (
            np.mean(filtered_data) if filtered_data else None
        )
    else:
        mean_KL_divergence_middle_2_from_front = None

    filtered_data = [x for x in KL_divergence_hind_2_from_front_list if x is not None]
    if filtered_data[0] != None:
        mean_KL_divergence_hind_2_from_front = (
            np.mean(filtered_data) if filtered_data else None
        )
    else:
        mean_KL_divergence_hind_2_from_front = None

    mean_x_axis = 1
    plt.figure(figsize=(10, 6))

    if mean_KL_divergence_hind_from_front is not None:
        plt.scatter(
            mean_x_axis,
            (mean_KL_divergence_middle_from_front),
            label="R2 from R1",
            color="blue",
        )
    if mean_KL_divergence_hind_from_front is not None:
        plt.scatter(
            mean_x_axis,
            (mean_KL_divergence_hind_from_front),
            label="R3 from R1",
            color="green",
        )

    if mean_KL_divergence_front_2_from_front is not None:
        plt.scatter(
            mean_x_axis,
            (mean_KL_divergence_front_2_from_front),
            label="L1 from R1",
            color="red",
        )

    if mean_KL_divergence_middle_2_from_front is not None:
        plt.scatter(
            mean_x_axis,
            (mean_KL_divergence_middle_2_from_front),
            label="L2 from R1",
            color="orange",
        )

    if mean_KL_divergence_hind_2_from_front is not None:
        plt.scatter(
            mean_x_axis,
            (mean_KL_divergence_hind_2_from_front),
            label="L3 from R1",
            color="purple",
        )

    plt.yscale(mscale.LogScale(axis="y", base=10))

    # Adding labels and title
    plt.xlabel("Trial Index")
    plt.ylabel("KL Divergence Value")
    plt.title(f"{args.tag} KL Divergence Across Trials")
    plt.legend(loc="best")

    # Show the plot
    plt.savefig(f"radius_mean/{args.tag}_KL_Divergence_radius.png")
