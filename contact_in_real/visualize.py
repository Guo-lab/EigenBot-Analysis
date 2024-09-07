import argparse
import itertools
import json
import test
from itertools import islice

import matplotlib.pyplot as plt
import matplotlib.scale as mscale
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy.stats import entropy
from tqdm import tqdm


def plot_time_based_histogram(
    new_01_10_length_series, new_01_length_series, new_10_length_series
):
    """
    This Function is deprecated now. Use plot_period_length_based_histogram instead.
    """
    new_01_length_series = np.repeat(new_01_length_series, 2)
    new_10_length_series = np.repeat(new_10_length_series, 2)
    x_01_10 = np.arange(len(new_01_10_length_series))
    x_01 = np.arange(len(new_01_length_series))
    x_10 = np.arange(len(new_10_length_series))
    plt.figure(figsize=(14, 6))
    plt.bar(
        x_01_10,
        new_01_10_length_series,
        label="01-10 Lengths",
        color="#008080",
        alpha=0.3,
    )
    plt.bar(x_01, new_01_length_series, label="01 Lengths", color="#DC143C", alpha=0.3)
    plt.bar(x_10, new_10_length_series, label="10 Lengths", color="#DAA520", alpha=0.3)

    # Labeling the plot
    plt.title("Index-Based Histogram of Phase Lengths")
    plt.xlabel("Index")
    plt.ylabel("Phase Length")
    plt.legend()
    # plt.show()


def plot_period_length_based_histogram(
    args, new_01_10_length_series, new_01_length_series, new_10_length_series, bins=10
):
    """
    Plot histograms for the 01_10, 01, and 10 length series.

    Parameters:
        new_01_10_length_series (np.array): Array representing the lengths of combined 01 and 10 transitions.
        new_01_length_series (np.array): Array representing the lengths of 01 transitions.
        new_10_length_series (np.array): Array representing the lengths of 10 transitions.
        bins (int): Number of bins to use for the histograms.
    """
    plt.figure(figsize=(18, 6))
    print()
    # print("Lengths of 01-10 Series: ", new_01_10_length_series)
    # print("Lengths of 01 Series: ", new_01_length_series)
    # print("Lengths of 10 Series: ", new_10_length_series)

    plt.subplot(1, 3, 1)
    plt.hist(
        new_01_10_length_series,
        bins=bins,
        color="blue",
        alpha=0.4,
    )
    plt.title("Histogram of 01-10 Length Series")
    plt.xlabel("Period of Phase. in Seconds")
    plt.ylabel("Data Frequency")

    plt.subplot(1, 3, 2)
    plt.hist(new_01_length_series, bins=bins, color="green", alpha=0.4)
    plt.title("Histogram of 01 Length Series")
    plt.xlabel("Period of Phase. in Seconds")
    plt.ylabel("Data Frequency")

    plt.subplot(1, 3, 3)
    plt.hist(new_10_length_series, bins=bins, color="red", alpha=0.4)
    plt.title("Histogram of 10 Length Series")
    plt.xlabel("Period of Phase. in Seconds")
    plt.ylabel("Data Frequency")
    plt.savefig(f"histplot/{args.group}_{args.leg_number}_hist.png")
    # plt.show()

    data = [new_01_10_length_series, new_01_length_series, new_10_length_series]
    labels = ["01-10 Length Series", "01 Length Series", "10 Length Series"]

    plt.figure(figsize=(9, 7))
    palette = sns.color_palette("Set2", n_colors=len(data))
    boxplot = plt.boxplot(data, vert=True, labels=labels, patch_artist=True)
    for patch, color in zip(boxplot["boxes"], palette):
        patch.set_facecolor(color)
        patch.set_edgecolor("black")  # Optionally set edge color
        patch.set_linewidth(1.5)

    plt.title("Box Plots of Length Series", fontsize=16, fontweight="bold")
    plt.xlabel("Length Series", fontsize=14)
    plt.ylabel("Period of Phase in Seconds", fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.savefig(f"boxplot/{args.group}_{args.leg_number}_boxplot.png")
    # plt.show()

    plt.close()


def plot_entropy_series(
    leg_number,
    file,
    zero_entropy_series,
    one_entropy_series,
    entropy_01_10_series,
    entropy_01_series,
    entropy_10_series,
):
    plt.figure(figsize=(12, 8))

    x_axis = np.arange(len(zero_entropy_series))

    plt.plot(x_axis, zero_entropy_series, label="Zero Entropy")  # marker="o"
    plt.plot(x_axis, one_entropy_series, label="One Entropy")  # marker="x"
    plt.plot(x_axis, entropy_01_10_series, label="01-10 Entropy")  # marker="s"
    plt.plot(x_axis, entropy_01_series, label="01 Entropy")  # marker="^"
    plt.plot(x_axis, entropy_10_series, label="10 Entropy")  # marker="v"

    plt.title("Sliding Window Entropy Analysis")
    plt.xlabel("Window Index")
    plt.ylabel("Entropy (bits)")
    plt.yscale(mscale.LogScale(axis="y", base=10))

    plt.legend()
    plt.savefig(f"entropy/{file}_{leg_number}_entropy_over_time.png")
    # plt.show()


def all_legs_hills(
    args,
    time_bins,
    swing_counts_front_r=None,
    swing_counts_middle_r=None,
    swing_counts_hind_r=None,
    swing_counts_front_l=None,
    swing_counts_middle_l=None,
    swing_counts_hind_l=None,
    if_prob=False,
):
    all_cnts = [
        swing_counts_front_l,
        swing_counts_middle_l,
        swing_counts_hind_l,
        swing_counts_front_r,
        swing_counts_middle_r,
        swing_counts_hind_r,
    ]
    filtered_cnts = [cnt for cnt in all_cnts if cnt is not None]
    fig, axs = plt.subplots(len(filtered_cnts), 1, figsize=(12, 9), sharex=True)
    if if_prob:
        common_ylim = (0, 1)
    else:
        common_ylim = (0, max([max(cnt) for cnt in filtered_cnts]))
    axs_i = 0
    axs[axs_i].bar(
        time_bins,
        swing_counts_front_l,
        width=0.06,
        color="blue",
        alpha=0.4,
        label="Left Front Leg",
    )
    axs[axs_i].set_ylim(common_ylim)
    axs[axs_i].set_title("Swing Phase Probability - Left Front Leg")
    axs[axs_i].set_ylabel("Swing Probability")
    axs[axs_i].legend()
    axs_i += 1
    if swing_counts_middle_l is not None:
        axs[axs_i].bar(
            time_bins,
            swing_counts_middle_l,
            width=0.06,
            color="lightblue",
            alpha=0.4,
            label="Left Middle Leg",
        )

        axs[axs_i].set_ylim(common_ylim)
        axs[axs_i].set_title("Swing Phase Probability - Left Middle Leg")
        axs[axs_i].set_ylabel("Swing Probability")
        axs[axs_i].legend()
        axs_i += 1
    if swing_counts_hind_l is not None:
        axs[axs_i].bar(
            time_bins,
            swing_counts_hind_l,
            width=0.06,
            color="darkblue",
            alpha=0.4,
            label="Left Hind Leg",
        )

        axs[axs_i].set_ylim(common_ylim)
        axs[axs_i].set_title("Swing Phase Probability - Left Hind Leg")
        axs[axs_i].set_ylabel("Swing Probability")
        axs[axs_i].legend()
        axs_i += 1
    axs[axs_i].bar(
        time_bins,
        swing_counts_front_r,
        width=0.06,
        color="red",
        alpha=0.4,
        label="Right Front Leg",
    )

    axs[axs_i].set_ylim(common_ylim)
    axs[axs_i].set_title("Swing Phase Probability - Right Front Leg")
    axs[axs_i].set_ylabel("Swing Probability")
    axs[axs_i].legend()
    axs_i += 1
    if swing_counts_middle_r is not None:
        axs[axs_i].bar(
            time_bins,
            swing_counts_middle_r,
            width=0.06,
            color="lightcoral",
            alpha=0.4,
            label="Right Middle Leg",
        )

        axs[axs_i].set_ylim(common_ylim)
        axs[axs_i].set_title("Swing Phase Probability - Right Middle Leg")
        axs[axs_i].set_ylabel("Swing Probability")
        axs[axs_i].legend()
        axs_i += 1
    if swing_counts_hind_r is not None:
        axs[axs_i].bar(
            time_bins,
            swing_counts_hind_r,
            width=0.06,
            color="darkred",
            alpha=0.4,
            label="Right Hind Leg",
        )

        axs[axs_i].set_ylim(common_ylim)
        axs[axs_i].set_title("Swing Phase Probability - Right Hind Leg")
        axs[axs_i].set_ylabel("Swing Probability")
        axs[axs_i].legend()
        axs_i += 1

    plt.tight_layout()
    plt.savefig(f"prob_all_legs/{args.group}_all_legs_swing_phase_combined.png")
    # plt.show()


def double_hill_plot(
    args,
    time_bins,
    swing_counts_front=None,
    swing_counts_middle=None,
    swing_counts_hind=None,
):
    bar_width = 0.06
    positions_front = time_bins - bar_width
    positions_hind = time_bins
    fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    bars_front = axs[0].bar(
        positions_front,
        swing_counts_front,
        width=bar_width,
        color="blue",
        alpha=0.4,
        label=f"{args.side.capitalize()} Front Leg",
    )
    if swing_counts_hind is not None:
        bars_hind = axs[0].bar(
            positions_hind,
            swing_counts_hind,
            width=bar_width,
            color="purple",
            alpha=0.4,
            label=f"{args.side.capitalize()} Hind Leg",
        )
    else:
        bars_hind = None

    axs[0].set_title(
        f"Swing Phase Frequency - {args.side.capitalize()} Front Leg. {args.side.capitalize()} Hind Leg"
    )
    axs[0].set_ylabel("Swing Probability")
    axs[0].legend()

    if swing_counts_middle is None:
        return fig, axs, bars_front, bars_hind, None
    bars_middle = axs[1].bar(
        time_bins,
        swing_counts_middle,
        width=0.06,
        color="red",
        alpha=0.4,
        label=f"{args.side.capitalize()} Middle Leg",
    )
    axs[1].set_title(f"Swing Phase Frequency - {args.side.capitalize()} Middle Leg")
    axs[1].set_ylabel("Swing Probability")
    axs[1].legend()

    plt.tight_layout()
    # plt.savefig(f"{args.group}_swing_phase_combined.png")
    # plt.show()

    return fig, axs, bars_front, bars_hind, bars_middle


def double_hill_incr_plot(
    args,
    time_bins,
    swing_counts_front=None,
    swing_counts_middle=None,
    swing_counts_hind=None,
    swing_counts_front_2=None,
    swing_counts_middle_2=None,
    swing_counts_hind_2=None,
):
    bar_width = 0.075
    positions_front = time_bins - bar_width
    positions_hind = time_bins

    fig, axs = plt.subplots(2, 1, figsize=(10, 4), sharex=True)
    y_limit = (0, 0.6)
    # In NAME_LIST_7: switch R3 and L3

    # Dark Red: #C0392B # Teal
    # Orange: #E67E22
    # Light Yellow: #F1C40F
    # Cold Colors:
    # Dark Blue: #2980B9
    # Teal: #16A085
    # Light Blue: #85C1E9

    # Warm 1: #FF5733 (Red-Orange, 80% opacity)
    # Warm 2: #FFC300 (Golden Yellow, 80% opacity)
    # Warm 3: #FF6F61 (Coral, 80% opacity)
    # Cold 1: #3498DB (Bright Blue, 80% opacity)
    # Cold 2: #1ABC9C (Turquoise, 80% opacity)
    # Cold 3: #5DADE2 (Sky Blue, 80% opacity)

    raw_colors = {
        "cold_2": "#1ABC9C",
        "warm_2": "#FF5733",
        #
        "cold_1": "#00A2E8",
        "warm_3": "#F1C40F",  # "#E67E22",  # "#FF6F61",
        #
        "cold_3": "purple",
        "warm_1": "#E67E22",  # "#FFC300",
    }
    colors = [  # In NAME_LIST_7: switch R3 and L3
        raw_colors["cold_2"],  # Right Front
        raw_colors["warm_2"],  # Left Front
        #
        raw_colors["cold_1"],  # Right Hind
        raw_colors["warm_3"],  # Left Hind
        #
        raw_colors["cold_3"],  # Left Middle
        raw_colors["warm_1"],  # Right Middle
    ]
    # colors = [
    #     "blue",  # Right Front
    #     "red",  # Left Front
    #     "green",  # Right Hind
    #     "orange",  # Left Hind
    #     "darkblue",  # Left Middle
    #     "yellow",  # Right Middle
    # ]
    labels = [
        "R1",  # "Right Front Leg",
        "L1",  # "Left Front Leg",
        "R3",  # "Right Hind Leg",
        "L3",  # "Left Hind Leg",
        "L2",  # "Left Middle Leg",
        "R2",  # "Right Middle Leg",
    ]
    alphas = [0.8, 0.8, 0.8, 0.8, 0.5, 0.8]
    if args.group == "NAME_LIST_7":
        bars_front = axs[0].bar(
            positions_front,
            swing_counts_front,
            width=bar_width,
            color=colors[0],
            alpha=alphas[0],
            label=labels[0],
        )
        if swing_counts_hind_2 is not None:
            bars_hind_2 = axs[0].bar(
                positions_hind,
                swing_counts_hind_2,
                width=bar_width,
                color=colors[3],
                alpha=alphas[3],
                label=labels[3],
            )
        else:
            bars_hind_2 = None

        bars_front_2 = axs[1].bar(
            positions_front,
            swing_counts_front_2,
            width=bar_width,
            color=colors[1],
            alpha=alphas[1],
            label=labels[1],
        )
        if swing_counts_hind is not None:
            bars_hind = axs[1].bar(
                positions_hind,
                swing_counts_hind,
                width=bar_width,
                color=colors[2],
                alpha=alphas[2],
                label=labels[2],
            )
        else:
            bars_hind = None
        bars_middle, bars_middle_2 = None, None

    elif args.group == "NAME_LIST_5":
        bars_front = axs[0].bar(
            positions_front,
            swing_counts_front,
            width=bar_width,
            color=colors[0],
            alpha=alphas[0],
            label=labels[0],
        )
        if swing_counts_hind_2 is not None:
            bars_hind_2 = axs[0].bar(
                positions_hind,
                swing_counts_hind_2,
                width=bar_width,
                color=colors[3],
                alpha=alphas[3],
                label=labels[3],
            )
        else:
            bars_hind_2 = None
        if swing_counts_middle_2 is not None:
            bars_middle_2 = axs[0].bar(
                positions_hind + bar_width,
                swing_counts_middle_2,
                width=bar_width,
                color=colors[4],
                alpha=alphas[4],
                label=labels[4],
            )

        bars_front_2 = axs[1].bar(
            positions_front,
            swing_counts_front_2,
            width=bar_width,
            color=colors[1],
            alpha=alphas[1],
            label=labels[1],
        )
        if swing_counts_hind is not None:
            bars_hind = axs[1].bar(
                positions_hind,
                swing_counts_hind,
                width=bar_width,
                color=colors[2],
                alpha=alphas[2],
                label=labels[2],
            )
        else:
            bars_hind = None
        bars_middle = None

    # elif args.group == "NAME_LIST_4": # It Looks Good to Me.
    #     pass

    else:
        bars_front = axs[0].bar(
            positions_front,
            swing_counts_front,
            width=bar_width,
            color=colors[0],
            alpha=alphas[0],
            label=labels[0],
        )

        if swing_counts_hind is not None:
            bars_hind = axs[0].bar(
                positions_hind,
                swing_counts_hind,
                width=bar_width,
                color=colors[2],
                alpha=alphas[2],
                label=labels[2],
            )
        else:
            bars_hind = None

        if swing_counts_middle_2 is not None:
            bars_middle_2 = axs[0].bar(
                positions_hind + bar_width,
                swing_counts_middle_2,
                width=bar_width,
                color=colors[4],
                alpha=alphas[4],
                label=labels[4],
            )
        else:
            bars_middle_2 = None

        bars_front_2 = axs[1].bar(
            positions_front,
            swing_counts_front_2,
            width=bar_width,
            color=colors[1],
            alpha=alphas[1],
            label=labels[1],
        )
        if swing_counts_hind_2 is not None:
            bars_hind_2 = axs[1].bar(
                positions_hind,
                swing_counts_hind_2,
                width=bar_width,
                color=colors[3],
                alpha=alphas[3],
                label=labels[3],
            )
        else:
            bars_hind_2 = None

        if swing_counts_middle is not None:
            bars_middle = axs[1].bar(
                positions_hind + bar_width,
                swing_counts_middle,
                width=bar_width,
                color=colors[5],
                alpha=alphas[5],
                label=labels[5],
            )
        else:
            bars_middle = None

    if bars_middle is None:
        axs[0].set_title(f"Swing Phase Probability For Right Middle Leg Amputation")
        if bars_middle_2 is None:
            axs[0].set_title(f"Swing Phase Probability For Both Middle Legs Amputation")
    elif bars_hind_2 is None:
        axs[0].set_title(f"Swing Phase Probability For Left Hind Leg Amputation")
        if bars_hind is None:
            axs[0].set_title(f"Swing Phase Probability For Both Hind Legs Amputation")
    else:
        axs[0].set_title(f"Swing Phase Probability 6 Legs")

    axs[0].set_ylabel("Swing Probability")
    axs[0].set_ylim(y_limit)
    axs[0].legend()

    axs[1].set_ylabel("Swing Probability")
    axs[1].set_ylim(y_limit)
    axs[1].legend()

    plt.tight_layout()
    return (
        fig,
        axs,
        bars_front,
        bars_middle,
        bars_hind,
        bars_front_2,
        bars_middle_2,
        bars_hind_2,
    )


def update_plot_incr(
    i,
    swing_counts_front,
    swing_counts_middle,
    swing_counts_hind,
    swing_counts_front_2,
    swing_counts_middle_2,
    swing_counts_hind_2,
    bars_front,
    bars_middle,
    bars_hind,
    bars_front_2,
    bars_middle_2,
    bars_hind_2,
):
    for bar, height in zip(bars_front, swing_counts_front[i]):
        bar.set_height(height)
    if bars_hind is not None:
        for bar, height in zip(bars_hind, swing_counts_hind[i]):
            bar.set_height(height)
    if bars_middle is not None:
        for bar, height in zip(bars_middle, swing_counts_middle[i]):
            bar.set_height(height)

    for bar, height in zip(bars_front_2, swing_counts_front_2[i]):
        bar.set_height(height)
    if bars_hind_2 is not None:
        for bar, height in zip(bars_hind_2, swing_counts_hind_2[i]):
            bar.set_height(height)
    if bars_middle_2 is not None:
        for bar, height in zip(bars_middle_2, swing_counts_middle_2[i]):
            bar.set_height(height)

    return bars_front, bars_middle, bars_hind, bars_front_2, bars_middle_2, bars_hind_2


def create_animation_incr(
    args,
    time_bins,
    gif_front_swing_prob,
    gif_middle_swing_prob,
    gif_hind_swing_prob,
    gif_front_swing_prob_2,
    gif_middle_swing_prob_2,
    gif_hind_swing_prob_2,
):
    (
        fig,
        axs,
        bars_front,
        bars_middle,
        bars_hind,
        bars_front_2,
        bars_middle_2,
        bars_hind_2,
    ) = double_hill_incr_plot(
        args,
        time_bins,
        gif_front_swing_prob[0],
        gif_middle_swing_prob[0] if len(gif_middle_swing_prob) != 0 else None,
        gif_hind_swing_prob[0] if len(gif_hind_swing_prob) != 0 else None,
        gif_front_swing_prob_2[0],
        gif_middle_swing_prob_2[0] if len(gif_middle_swing_prob_2) != 0 else None,
        gif_hind_swing_prob_2[0] if len(gif_hind_swing_prob_2) != 0 else None,
    )

    anim = FuncAnimation(
        fig,
        update_plot_incr,
        frames=len(gif_front_swing_prob),
        fargs=(
            gif_front_swing_prob,
            gif_middle_swing_prob,
            gif_hind_swing_prob,
            gif_front_swing_prob_2,
            gif_middle_swing_prob_2,
            gif_hind_swing_prob_2,
            bars_front,
            bars_middle,
            bars_hind,
            bars_front_2,
            bars_middle_2,
            bars_hind_2,
        ),
        blit=False,
    )

    # anim.save(
    #     f"incr/{args.group}_incr_swing_phase_combined.gif",
    #     writer=PillowWriter(fps=10),
    # )
    # anim.save(
    #     f"paper/{args.group}_incr_swing_phase_combined.gif",
    #     writer=PillowWriter(fps=10),
    # )
    plt.show()
    # plt.close()


def update_plot(
    i,
    swing_counts_front,
    swing_counts_middle,
    swing_counts_hind,
    bars_front,
    bars_hind,
    bars_middle,
):
    for bar, height in zip(bars_front, swing_counts_front[i]):
        bar.set_height(height)
    if bars_hind is not None:
        for bar, height in zip(bars_hind, swing_counts_hind[i]):
            bar.set_height(height)
    if bars_middle is not None:
        for bar, height in zip(bars_middle, swing_counts_middle[i]):
            bar.set_height(height)
    return bars_front, bars_hind, bars_middle


def create_animation(
    args, time_bins, gif_front_swing_prob, gif_middle_swing_prob, gif_hind_swing_prob
):
    if len(gif_middle_swing_prob) == 0:
        fig, axs, bars_front, bars_hind, bars_middle = double_hill_plot(
            args,
            time_bins,
            swing_counts_front=gif_front_swing_prob[0],
            swing_counts_hind=gif_hind_swing_prob[0],
        )
    elif len(gif_hind_swing_prob) == 0:
        fig, axs, bars_front, bars_hind, bars_middle = double_hill_plot(
            args,
            time_bins,
            swing_counts_front=gif_front_swing_prob[0],
            swing_counts_middle=gif_middle_swing_prob[0],
        )
    else:
        fig, axs, bars_front, bars_hind, bars_middle = double_hill_plot(
            args,
            time_bins,
            gif_front_swing_prob[0],
            gif_middle_swing_prob[0],
            gif_hind_swing_prob[0],
        )

    anim = FuncAnimation(
        fig,
        update_plot,
        frames=len(gif_front_swing_prob),
        fargs=(
            gif_front_swing_prob,
            gif_middle_swing_prob,
            gif_hind_swing_prob,
            bars_front,
            bars_hind,
            bars_middle,
        ),
        blit=False,
    )

    anim.save(
        f"{args.group}_{args.side}_swing_phase_combined.gif",
        writer=PillowWriter(fps=10),
    )
    # plt.show()
    plt.close()


def distance_divergence_plot(
    args,
    KL_divergence_middle_from_front,
    KL_divergence_hind_from_front,
    wasserstein_distance_front_middle,
    wasserstein_distance_front_hind,
    wasserstein_distance_middle_hind,
):
    plt.figure(figsize=(12, 8))

    x_len = np.max(
        [
            len(KL_divergence_middle_from_front),
            len(KL_divergence_hind_from_front),
            len(wasserstein_distance_front_middle),
            len(wasserstein_distance_front_hind),
            len(wasserstein_distance_middle_hind),
        ]
    )
    x_axis = np.arange(x_len)
    print(
        len(x_axis),
        len(KL_divergence_middle_from_front),
        len(KL_divergence_hind_from_front),
        len(wasserstein_distance_front_middle),
        len(wasserstein_distance_front_hind),
        len(wasserstein_distance_middle_hind),
    )

    if len(KL_divergence_middle_from_front) != 0:
        plt.plot(
            x_axis,
            KL_divergence_middle_from_front,
            label="KL Divergence Middle from Front",
        )
    if len(KL_divergence_hind_from_front) != 0:
        plt.plot(
            x_axis, KL_divergence_hind_from_front, label="KL Divergence Hind from Front"
        )
    if len(wasserstein_distance_front_middle) != 0:
        plt.plot(
            x_axis,
            wasserstein_distance_front_middle,
            label="Wasserstein Distance Front-Middle",
        )
    if len(wasserstein_distance_front_hind) != 0:
        plt.plot(
            x_axis,
            wasserstein_distance_front_hind,
            label="Wasserstein Distance Front-Hind",
        )
    if len(wasserstein_distance_middle_hind) != 0:
        plt.plot(
            x_axis,
            wasserstein_distance_middle_hind,
            label="Wasserstein Distance Middle-Hind",
        )  # marker="v"

    # print(
    #     KL_divergence_middle_from_front,
    #     "\n",
    #     KL_divergence_hind_from_front,
    #     "\n",
    #     wasserstein_distance_front_middle,
    #     "\n",
    #     wasserstein_distance_front_hind,
    #     "\n",
    #     wasserstein_distance_middle_hind,
    # )
    plt.title("Distance Divergence Analysis")
    plt.xlabel("Window Index")
    plt.ylabel("Distance")
    plt.yscale(mscale.LogScale(axis="y", base=10))

    plt.legend()
    plt.savefig(f"distance_divergence/{args.group}_{args.side}_distance_divergence.png")
    # plt.show()


def distance_plot(
    args,
    wasserstein_distance_front_middle,
    wasserstein_distance_front_hind,
    wasserstein_distance_middle_hind,
):
    plt.figure(figsize=(12, 8))

    x_len = np.max(
        [
            len(wasserstein_distance_front_middle),
            len(wasserstein_distance_front_hind),
            len(wasserstein_distance_middle_hind),
        ]
    )
    x_axis = np.arange(x_len)
    print(
        len(x_axis),
        len(wasserstein_distance_front_middle),
        len(wasserstein_distance_front_hind),
        len(wasserstein_distance_middle_hind),
    )

    if len(wasserstein_distance_front_middle) != 0:
        plt.plot(
            x_axis,
            wasserstein_distance_front_middle,
            label="Wasserstein Distance Front-Middle",
        )
    if len(wasserstein_distance_front_hind) != 0:
        plt.plot(
            x_axis,
            wasserstein_distance_front_hind,
            label="Wasserstein Distance Front-Hind",
        )
    if len(wasserstein_distance_middle_hind) != 0:
        plt.plot(
            x_axis,
            wasserstein_distance_middle_hind,
            label="Wasserstein Distance Middle-Hind",
        )  # marker="v"

    plt.title("Distance Analysis")
    plt.xlabel("Window Index")
    plt.ylabel("Distance")
    plt.yscale(mscale.LogScale(axis="y", base=10))

    plt.legend()
    plt.savefig(f"distance/{args.group}_{args.side}_distance.png")
    # plt.show()


def divergence_plot(
    args,
    KL_divergence_middle_from_front,
    KL_divergence_hind_from_front,
):
    plt.figure(figsize=(12, 8))

    x_len = np.max(
        [
            len(KL_divergence_middle_from_front),
            len(KL_divergence_hind_from_front),
        ]
    )
    x_axis = np.arange(x_len)
    print(
        len(x_axis),
        len(KL_divergence_middle_from_front),
        len(KL_divergence_hind_from_front),
    )

    if len(KL_divergence_middle_from_front) != 0:
        plt.plot(
            x_axis,
            KL_divergence_middle_from_front,
            label="KL Divergence Middle from Front",
        )
    if len(KL_divergence_hind_from_front) != 0:
        plt.plot(
            x_axis, KL_divergence_hind_from_front, label="KL Divergence Hind from Front"
        )

    plt.title("Divergence Analysis")
    plt.xlabel("Window Index")
    plt.ylabel("Divergence")
    plt.yscale(mscale.LogScale(axis="y", base=10))

    plt.legend()
    plt.savefig(f"divergence/{args.group}_{args.side}_divergence.png")
    # plt.show()


def divergence_incr_plot(
    args,
    KL_divergence_middle_from_front,
    KL_divergence_hind_from_front,
    KL_divergence_front_2_from_front,
    KL_divergence_middle_2_from_front,
    KL_divergence_hind_2_from_front,
):
    plt.figure(figsize=(8, 5))

    x_len = np.max(
        [
            len(KL_divergence_middle_from_front),
            len(KL_divergence_hind_from_front),
            len(KL_divergence_front_2_from_front),
            len(KL_divergence_middle_2_from_front),
            len(KL_divergence_hind_2_from_front),
        ]
    )
    x_axis = np.arange(x_len)
    print(
        len(x_axis),
        len(KL_divergence_middle_from_front),
        len(KL_divergence_hind_from_front),
    )

    if len(KL_divergence_middle_from_front) != 0:
        KL_divergence_middle_from_front = np.array(
            [np.nan if v is None else v for v in KL_divergence_middle_from_front]
        )
        plt.plot(
            x_axis,
            KL_divergence_middle_from_front,
            label=(
                "KL Divergence R2 from R1"
                if not np.all(np.isnan(KL_divergence_middle_from_front))
                else ""
            ),
        )
    if len(KL_divergence_hind_from_front) != 0:
        KL_divergence_hind_from_front = np.array(
            [np.nan if v is None else v for v in KL_divergence_hind_from_front]
        )
        plt.plot(
            x_axis,
            KL_divergence_hind_from_front,
            label=(
                "KL Divergence R3 from R1"
                if not np.all(np.isnan(KL_divergence_hind_from_front))
                else ""
            ),
        )

    if len(KL_divergence_front_2_from_front) != 0:
        plt.plot(
            x_axis,
            KL_divergence_front_2_from_front,
            label="KL Divergence L1 from R1",
        )

    if len(KL_divergence_middle_2_from_front) != 0:
        KL_divergence_middle_2_from_front = np.array(
            [np.nan if v is None else v for v in KL_divergence_middle_2_from_front]
        )
        plt.plot(
            x_axis,
            KL_divergence_middle_2_from_front,
            label=(
                "KL Divergence L2 from R1"
                if not np.all(np.isnan(KL_divergence_middle_2_from_front))
                else ""
            ),
        )

    if len(KL_divergence_hind_2_from_front) != 0:
        KL_divergence_hind_2_from_front = np.array(
            [np.nan if v is None else v for v in KL_divergence_hind_2_from_front]
        )
        plt.plot(
            x_axis,
            KL_divergence_hind_2_from_front,
            color="black",
            label=(
                "KL Divergence L3 from R1"
                if not np.all(np.isnan(KL_divergence_hind_2_from_front))
                else ""
            ),
        )

    plt.title("Divergence Analysis")
    plt.xlabel("Window Index")
    plt.ylabel("Divergence")
    plt.yscale(mscale.LogScale(axis="y", base=10))

    plt.legend()

    plt.savefig(f"incr_divergence/{args.group}_incr_divergence.png")


def distance_incr_plot(
    args,
    wasserstein_distance_front_middle,
    wasserstein_distance_front_hind,
    wasserstein_distance_front_front_2,
    wasserstein_distance_front_middle_2,
    wasserstein_distance_front_hind_2,
):
    plt.figure(figsize=(12, 8))
    x_len = np.max(
        [
            len(wasserstein_distance_front_middle),
            len(wasserstein_distance_front_hind),
            len(wasserstein_distance_front_front_2),
            len(wasserstein_distance_front_middle_2),
            len(wasserstein_distance_front_hind_2),
        ]
    )
    x_axis = np.arange(x_len)
    print(
        len(x_axis),
        len(wasserstein_distance_front_middle),
        len(wasserstein_distance_front_hind),
    )

    if len(wasserstein_distance_front_middle) != 0:
        wasserstein_distance_front_middle = np.array(
            [
                np.nan if v is None or np.isnan(v) else v
                for v in wasserstein_distance_front_middle
            ]
        )
        plt.plot(
            x_axis,
            wasserstein_distance_front_middle,
            label=(
                "Wasserstein Distance R1 - R2"
                if not np.all(np.isnan(wasserstein_distance_front_middle))
                else ""
            ),
        )
    if len(wasserstein_distance_front_hind) != 0:
        wasserstein_distance_front_hind = np.array(
            [
                np.nan if v is None or np.isnan(v) else v
                for v in wasserstein_distance_front_hind
            ]
        )
        plt.plot(
            x_axis,
            wasserstein_distance_front_hind,
            label=(
                "Wasserstein Distance R1 - R3"
                if not np.all(np.isnan(wasserstein_distance_front_hind))
                else ""
            ),
        )
    if len(wasserstein_distance_front_front_2) != 0:
        plt.plot(
            x_axis,
            wasserstein_distance_front_front_2,
            label="Wasserstein Distance R1 - L1",
        )

    if len(wasserstein_distance_front_middle_2) != 0:
        wasserstein_distance_front_middle_2 = np.array(
            [
                np.nan if v is None or np.isnan(v) else v
                for v in wasserstein_distance_front_middle_2
            ]
        )
        plt.plot(
            x_axis,
            wasserstein_distance_front_middle_2,
            label=(
                "Wasserstein Distance R1 - L2"
                if not np.all(np.isnan(wasserstein_distance_front_middle_2))
                else ""
            ),
        )
    if len(wasserstein_distance_front_hind_2) != 0:
        wasserstein_distance_front_hind_2 = np.array(
            [
                np.nan if v is None or np.isnan(v) else v
                for v in wasserstein_distance_front_hind_2
            ]
        )
        plt.plot(
            x_axis,
            wasserstein_distance_front_hind_2,
            color="black",
            label=(
                "Wasserstein Distance R1 - L3"
                if not np.all(np.isnan(wasserstein_distance_front_hind_2))
                else ""
            ),
        )

    plt.title("Distance Analysis")
    plt.xlabel("Window Index")
    plt.ylabel("Distance")
    plt.yscale(mscale.LogScale(axis="y", base=10))

    plt.legend()
    # plt.savefig(f"incr_distance/{args.group}_incr_distance.png")
    # plt.show()
