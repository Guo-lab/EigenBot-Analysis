import argparse
import itertools
import json
import test
from itertools import islice

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import entropy
from tqdm import tqdm
from visualize import (plot_entropy_series, plot_period_length_based_histogram,
                       plot_time_based_histogram)


def get_cycle_span_to_series(data):
    """
    Params:
        data: phase binary data
    Returns:
        phase_labels: 0 for swing, and 1 for stance
        lengths: the length of the period of one phase (zeros or ones). This is the index-based instead of the real-timestamp-based.

        data: [0, 1, 1, 1, 0, 0]
        labels: [0, 1, 0]
        lengths: [1, 3, 2]
    """
    change_points = np.where(np.diff(data) != 0)[0] + 1

    groups = np.split(data, change_points)
    phase_labels = np.array([group[0] for group in groups])
    lengths = np.array([len(group) for group in groups])

    return phase_labels, lengths


def get_cycle_span_to_time_span_series(data):
    """
    Params:
        data: the DataFrame Data including all columns.
    Returns:
        phase_labels: 0 for swing, and 1 for stance
        time_span_lengths: the length of the period of one phase (zeros or ones). This is calculated from the real timestamp.

        binary_data: [0, 1, 1, 1, 0, 0]
        timestamps: [0.5, 2, 2,5, 4, 4.5, 5]
        labels: [0, 1, 0]
        time_span_lengths: [1.5, 2.5, 0.5]
    """
    binary_data = data["Phase"].to_numpy()
    timestamps = data["Timestamp"].to_numpy()
    change_points = np.where(np.diff(binary_data) != 0)[0] + 1
    change_points = np.concatenate(([0], change_points, [len(binary_data) - 1]))

    # Extract phases and sum the time differences within each phase segment
    phase_labels = binary_data[change_points[:-1]]
    time_span_lengths = [
        timestamps[end] - timestamps[start]
        for start, end in zip(change_points[:-1], change_points[1:])
    ]
    return phase_labels, time_span_lengths


def get_entropy_from_time_series(data):
    """
    Params:
        data: Time Series Data
    Returns:
        the entropy of the time series. Shannon Entropy
    """
    prob_distribution = data / data.sum()
    prob_distribution = prob_distribution[prob_distribution > 0]

    if entropy(prob_distribution, base=2) == 0:
        print(prob_distribution)
    return -np.sum(prob_distribution * np.log(prob_distribution) / np.log(2))


def is_sublist(sublist, data_list):
    """
    The Function to check if the sublist is the subset of the data_list.
        [1,2,3] belongs to [0,0,1,2,3,2,1]
    """
    sublist_len = len(sublist)
    return any(
        sublist == list(islice(data_list, i, i + sublist_len))
        for i in range(len(data_list) - sublist_len + 1)
    )


def write_to_json_to_plot_hist(
    args,
    new_010_time_span_length_series,
    new_01_time_span_length_series,
    new_10_time_span_length_series,
):
    json_file_path = f"{args.group}_{args.leg_number}_time_span_series.json"
    try:
        with open(json_file_path, "r") as json_file:
            data = json.load(json_file)
    except FileNotFoundError:
        data = {
            "new_010_time_span_length_series": [],
            "new_01_time_span_length_series": [],
            "new_10_time_span_length_series": [],
        }

    if not is_sublist(
        new_010_time_span_length_series, data["new_010_time_span_length_series"]
    ):
        # print(f"{len(new_010_time_span_length_series)} Data Points Added to JSON 010")
        data["new_010_time_span_length_series"].extend(new_010_time_span_length_series)
        data["new_01_time_span_length_series"].extend(new_01_time_span_length_series)
        data["new_10_time_span_length_series"].extend(new_10_time_span_length_series)

    with open(json_file_path, "w") as json_file:
        json.dump(data, json_file, indent=4)


def calculate_gait_entropies(
    args,
    leg_data,
    full_leg_data,
    histogram=False,
):
    """
    This function will calculate the entropy data for a time series of leg data phase.

    Parameters:
        leg_data: numpy array of leg data phase, 0 for swing and 1 for stance. Data from the internal state of the real EigenBot.
        full_leg_data: the DataFrame for each leg. Including all columns.

    Returns:
        zero_entropy:   Entropy of the time series of swing phase lengths.
        one_entropy:    Entropy of the time series of stance phase lengths.
        entropy_01_10:  Entropy of the time series of 010/101 phase lengths.
        entropy_01:     Entropy of the time series of 01 phase lengths.
        entropy_10:     Entropy of the time series of 10 phase lengths.
    """
    # phase_labels_, lengths_ = get_cycle_span_to_series(leg_data)  # index-based
    phase_labels, time_span_lengths = get_cycle_span_to_time_span_series(full_leg_data)
    assert len(phase_labels) == len(
        time_span_lengths
    ), "length labels should match length of the phase period"
    time_span_lengths = np.array(time_span_lengths)

    zero_entropy = get_entropy_from_time_series(
        time_span_lengths[np.where(phase_labels == 0)[0]]
    )
    one_entropy = get_entropy_from_time_series(
        time_span_lengths[np.where(phase_labels == 1)[0]]
    )

    last_group = None
    new_010_time_span_length_series = []
    new_01_time_span_length_series = []
    new_10_time_span_length_series = []

    identical_groups = [
        list(group) for _, group in itertools.groupby(full_leg_data["Phase"])
    ]
    for each_group in identical_groups:  # all 10101010...
        if last_group:
            if last_group[0] == 0 and each_group[0] == 1:
                new_01_time_span_length_series.append(
                    time_span_lengths[identical_groups.index(each_group)]
                    + time_span_lengths[identical_groups.index(each_group) - 1]
                )
            if last_group[0] == 1 and each_group[0] == 0:
                new_10_time_span_length_series.append(
                    time_span_lengths[identical_groups.index(each_group)]
                    + time_span_lengths[identical_groups.index(each_group) - 1]
                )
        last_group = each_group

    for i in range(2, len(identical_groups)):
        if (
            identical_groups[i - 2][0] == 0
            and identical_groups[i - 1][0] == 1
            and identical_groups[i][0] == 0
        ):
            time_span_010 = (
                time_span_lengths[i]
                + time_span_lengths[i - 1]
                + time_span_lengths[i - 2]
            )
            new_010_time_span_length_series.append(time_span_010)

    write_to_json_to_plot_hist(
        args,
        new_010_time_span_length_series,
        new_01_time_span_length_series,
        new_10_time_span_length_series,
    )

    new_010_time_span_length_series = np.array(new_010_time_span_length_series)
    new_01_time_span_length_series = np.array(new_01_time_span_length_series)
    new_10_time_span_length_series = np.array(new_10_time_span_length_series)

    histogram = False
    if histogram:  # histogram is the verbose to debug for each secnario in one group
        """
        The following code is deprecated
        (not what we want any more, see plot_period_length_based_histogram)
        """
        # plot_time_based_histogram(
        #     new_01_10_length_series, new_01_length_series, new_10_length_series
        # )
        plot_period_length_based_histogram(
            args,
            new_010_time_span_length_series,
            new_01_time_span_length_series,
            new_10_time_span_length_series,
        )

    entropy_01_10 = get_entropy_from_time_series(new_010_time_span_length_series)
    entropy_01 = get_entropy_from_time_series(new_01_time_span_length_series)
    entropy_10 = get_entropy_from_time_series(new_10_time_span_length_series)

    return (
        np.array(zero_entropy),
        np.array(one_entropy),
        np.array(entropy_01_10),
        np.array(entropy_01),
        np.array(entropy_10),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="as the file label")
    parser.add_argument(
        "--file",
        type=str,
        default="",
        help="The name of the file to read to calculate the entropy and histogram",
    )
    parser.add_argument(
        "--leg_number",
        type=int,
        default=17,
        help="The leg number to calculate the entropy and histogram",
    )
    parser.add_argument(
        "--group",
        type=str,
        default="",
        help="The group of the data to calculate the entropy and histogram",
    )
    args = parser.parse_args()
    print(args)

    df_all = pd.read_csv(f"{args.file}_data.csv")
    # print(f"=====================READ Data: \n{df_all}\n======================\n")

    leg_17_data = df_all[df_all["Leg"] == args.leg_number]["Phase"].to_numpy()
    leg_data = df_all[df_all["Leg"] == args.leg_number]
    # print("Legs' phase data: ", leg_data)
    # print("Legs' phase data len: ", len(leg_data))
    print(
        f"Time Range: {df_all['Timestamp'].min()} - {df_all['Timestamp'].max()};",
        f"{df_all['Timestamp'].max() - df_all['Timestamp'].min()}",
    )

    zero_entropy, one_entropy, entropy_01_10, entropy_01, entropy_10 = (
        calculate_gait_entropies(args, leg_17_data, leg_data, histogram=True)
    )
    print("================Entropy Analysis on One Run:")
    print(f"Entropy of zero lengths: {zero_entropy:.4f} bits")
    print(f"Entropy of one lengths: {one_entropy:.4f} bits")
    print(f"Entropy of 01 10 gait lengths: {entropy_01_10:.4f} bits")
    print(f"Entropy of 01 gait lengths: {entropy_01:.4f} bits")
    print(f"Entropy of 10 gait lengths: {entropy_10:.4f} bits")
    print()

    window_size = 100
    step = 15

    zero_entropies = []
    one_entropies = []
    entropy_01_10s = []
    entropy_01s = []
    entropy_10s = []

    print("Entropy Analysis on Sliding Window:")
    for i in tqdm(
        range(0, len(leg_17_data) - window_size + 1, step), desc="Calculating Entropies"
    ):
        window_data = leg_17_data[i : i + window_size]
        window_full_data = leg_data[i : i + window_size]
        zero_entropy, one_entropy, entropy_01_10, entropy_01, entropy_10 = (
            calculate_gait_entropies(
                args, window_data, window_full_data, histogram=True
            )
        )

        zero_entropies.append(zero_entropy)
        one_entropies.append(one_entropy)
        entropy_01_10s.append(entropy_01_10)
        entropy_01s.append(entropy_01)
        entropy_10s.append(entropy_10)

        tmp_tuple = zero_entropy, one_entropy, entropy_01_10, entropy_01, entropy_10

    plot_entropy_series(
        args.leg_number,
        args.file,
        zero_entropies,
        one_entropies,
        entropy_01_10s,
        entropy_01s,
        entropy_10s,
    )
    print()
