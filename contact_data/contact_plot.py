import argparse

import matplotlib.pyplot as plt
import pandas as pd

parser = argparse.ArgumentParser(description="as the file label")

parser.add_argument(
    "--terrain",
    type=str,
    default="",
    help="terrain the robot is walking on: hill or flat",
)
parser.add_argument(
    "--control", type=str, default="", help="control method used: predefined or neural"
)
parser.add_argument(
    "--curve",
    type=bool,
    default=False,
    help="Whether do curve walking (True or False).",
)
parser.add_argument("--curve_method", type=str, default="", help="method one or two")
parser.add_argument(
    "--comparison", type=str, default="", help="what we want to compare: ipsi or contra"
)
parser.add_argument(
    "--data_type",
    type=str,
    default="",
    help="what data we want to plot: contact or phase",
)

args = parser.parse_args()
print(args)


def moving_window_classification(df, leg_index, window_size=10):
    """
    Classify each timestamp as 0 or 1 based on the majority value within a window around each timestamp.

    Parameters:
    - df: DataFrame with 'Timestamp' and 'Contact_1' columns.
    - window_size: The size of the window for classification.

    Returns:
    - A Series with the classified values.
    """
    classified = []
    print(leg_index)
    for i in range(len(df)):
        start_time = df.iloc[i]["Timestamp"] - window_size // 2
        end_time = df.iloc[i]["Timestamp"] + window_size // 2

        # Filter data within the window
        window_data = df[
            (df["Timestamp"] >= start_time) & (df["Timestamp"] <= end_time)
        ]

        # Determine the majority value in the window
        majority_value = window_data[f"Contact_{leg_index + 1}"].mode().iloc[0]
        classified.append(majority_value)

    return pd.Series(classified, index=df.index)


# 'phase_terrain_predefined' # 'terrain_predefined'
# 'phase_terrain_neural' # 'terrain_neural'
# 'phase_flat_predefined' # 'flat_predefined'
# 'phase_flat_neural' # 'flat_neural'
# suffix = 'terrain_predefined'

suffix = f"{args.terrain}_{args.control}"
print(f"Plotting {suffix} data under comparison {args.comparison}.")

prefix = args.data_type

if args.curve:
    print(f"Doing curve walking with method {args.curve_method}.")
    suffix = f"curve_{args.curve_method}"

leg_mapping = None
if args.comparison == "ipsi":
    # colors = ['#1f77b4',   # Left Front (Dark and Light)
    #         '#ff7f0e', # Left Middle (Dark and Light)
    #         '#2ca02c', # Left Hind (Dark and Light)
    #         '#4c99d4',
    #         '#ff9f30',
    #         '#76c7a2']
    colors = ["#000000"] * 6

    leg_names = [
        "Left Front",
        "Left Middle",
        "Left Hind",
        "Right Front",
        "Right Middle",
        "Right Hind",
    ]

    short_leg_names = ["L1", "L2", "L3", "R1", "R2", "R3"]

elif args.comparison == "contra":
    # colors = ['#1f77b4', '#4c99d4',  # Left Front (Dark and Light)
    #             '#ff7f0e', '#ff9f30',  # Left Middle (Dark and Light)
    #             '#2ca02c', '#76c7a2']  # Left Hind (Dark and Light)
    colors = ["#000000"] * 6

    leg_names = [
        "Left Front",
        "Right Front",
        "Left Middle",
        "Right Middle",
        "Left Hind",
        "Right Hind",
    ]

    short_leg_names = ["LF", "RF", "LM", "RM", "LH", "RH"]

else:
    print("Invalid comparison. Please choose either ipsi or contra.")
    exit()


leg_mapping = {
    "Left Front": 0,
    "Left Middle": 1,
    "Left Hind": 2,
    "Right Front": 3,
    "Right Middle": 4,
    "Right Hind": 5,
}


df = pd.read_csv(f"{args.data_type}_data_{suffix}.csv")

plt.figure(figsize=(21, 5))
for i, leg_name in enumerate(leg_names):
    leg_index = leg_mapping[leg_name]
    print(i, leg_name, leg_index)

    axs = plt.subplot(6, 1, i + 1)

    height_scale = 1
    df[f"Contact_{leg_index + 1}"] = moving_window_classification(
        df, leg_index, window_size=2
    )

    plt.fill_between(
        df["Timestamp"],
        0,
        1 * height_scale,
        where=df[f"Contact_{leg_index + 1}"] == 1,
        color="white",
    )
    df = df.sort_values("Timestamp")

    # print(df.loc[df[f"Contact_{leg_index + 1}"] == 0].count())
    # print(df[df[f"Contact_{leg_index + 1}"] == 0])
    plt.fill_between(
        df["Timestamp"],
        0,
        height_scale,
        color=colors[i],
        alpha=0.75,
        where=df[f"Contact_{leg_index + 1}"] == 0,
    )
    plt.plot(
        df["Timestamp"],
        df[f"Contact_{leg_index + 1}"] * height_scale,
        "o",
        label="Contact State",
    )

    # print(df[f"Contact_{leg_index + 1}"])
    if "curve_walking_stage" in df.columns and args.curve:
        for j in range(1, 3):
            plt.fill_between(
                df["Timestamp"],
                0,
                height_scale,
                where=(df["curve_walking_stage"] == j),
                color="gray",
                alpha=0.3,
            )

    y_max = 1
    plt.yticks([0, y_max])
    plt.yticks([])
    plt.ylim(0, y_max)
    plt.xlim(df["Timestamp"].min(), df["Timestamp"].max())
    if i == 5:
        plt.xticks(rotation=45, fontsize=8)

    if i < len(leg_names) - 1:
        plt.xticks([], fontsize=8)

    # plt.title(f'{leg_name} Leg')
    plt.ylabel(f"{short_leg_names[i]}", fontsize=12, rotation=90, labelpad=7)
    # axs.text(50, 0.0, f'{leg_name} Leg', fontsize=10, ha='center', va='top', rotation=0)

    # plt.grid(True)

plt.xlabel("Timestamp [s]", labelpad=-3)
plt.gcf().text(
    0.02,
    0.5,
    "Contact" if args.data_type == "contact" else "Phase",
    va="center",
    ha="center",
    rotation="vertical",
    fontsize=8,
)

# handles = [
#     plt.Line2D([0], [0],
#                color=colors[i], lw=5
#             ) for i in range(len(leg_names))]

# legends = [f"{leg_name} Leg" for leg_name in leg_names]
# plt.legend(
#     handles,
#     legends,
#     fontsize=12,
#     # loc='upper right',
#     bbox_to_anchor=(1.0, 7)
# )

# plt.tight_layout()
plt.subplots_adjust(left=0.05, right=0.98, hspace=0.8)
if args.data_type == "contact":
    plt.suptitle("Contact Data from 6 Legs in the EigenBot", fontsize=12, y=0.95)

elif args.data_type == "phase":
    plt.suptitle("Phase Data from 6 Legs in the EigenBot", fontsize=12, y=0.95)

else:
    print("Invalid data type. Please choose either contact or phase.")

import os

directory = f"plot_{suffix}"
if not os.path.exists(directory):
    os.makedirs(directory)
    print(f"Directory '{directory}' created.")

plt.savefig(f"{directory}/{args.comparison}_{args.data_type}_{suffix}.png")
# plt.show()
