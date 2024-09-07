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
    "--comparison", type=str, default="", help="what we want to compare: ipsi or contra"
)
parser.add_argument(
    "--data_type",
    type=str,
    default="",
    help="what data we want to plot: contact or phase",
)
parser.add_argument(
    "--data_name",
    type=str,
    default="",
    help="name of the data we are plotting",
)


args = parser.parse_args()
df_all = pd.read_csv(f"{args.data_name}_data.csv")
print(args)

############################################
# ======== Constructing the Dict ========= #
############################################
if args.comparison == "ipsi":
    colors = [
        "#1f77b4",  # Left Front (Dark and Light)
        "#ff7f0e",  # Left Middle (Dark and Light)
        "#2ca02c",  # Left Hind (Dark and Light)
        "#4c99d4",
        "#ff9f30",
        "#76c7a2",
    ]
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
    colors = ["#000000"] * 6
    leg_names = [
        "Left Front",
        "Right Front",
        "Left Middle",
        "Right Middle",
        "Left Hind",
        "Right Hind",
    ]

else:
    print("Invalid comparison. Please choose either ipsi or contra.")
    exit()

leg_module_mapping = {
    "Left Front": 5,
    "Left Middle": 2,
    "Left Hind": 18,
    "Right Front": 13,
    "Right Middle": 19,
    "Right Hind": 17,
}


############################################
# ========== Plotting the Data =========== #
############################################
plt.figure(figsize=(21, 5))
for i, leg_name in enumerate(leg_names):
    leg_module = leg_module_mapping[leg_name]

    columns = ["Leg", "Timestamp", "Phase", "Contact"]
    df = pd.DataFrame(columns=columns)
    print(leg_module)
    df = df_all[df_all["Leg"] == leg_module]
    print(f"DataFrame for {leg_name} with module number {leg_module}: \n", df.head())
    if df.empty:
        print(f"No data found for {leg_name}.")
        continue

    axs = plt.subplot(6, 1, i + 1)

    #################################################################
    y_max = 1
    plt.yticks([0, y_max])
    plt.yticks([])
    plt.ylim(0, y_max)
    if i < len(leg_names) - 1:
        plt.xticks([], fontsize=8)

    plt.xlim(df["Timestamp"].min(), df["Timestamp"].max())
    if i == 5:
        plt.xticks(rotation=45, fontsize=8)
    #################################################################

    height_scale = 1
    if args.data_type == "phase":  ###### NOTE: Weird filling issue.
        plt.fill_between(
            df["Timestamp"],
            0,
            1 * height_scale,
            where=df["Phase"] == 0,
            color="white",
        )
        plt.fill_between(
            df["Timestamp"],
            df["Phase"] * height_scale,
            color=colors[i],
            alpha=0.75,
            where=df["Phase"] == 1,
        )

    elif args.data_type == "contact":
        plt.fill_between(
            df["Timestamp"],
            0,
            1 * height_scale,
            where=df["Contact"] == 0,
            color="white",
        )
        plt.fill_between(
            df["Timestamp"],
            df["Contact"] * height_scale,
            color=colors[i],
            alpha=0.75,
            where=df["Contact"] == 1,
        )

    else:
        print("Invalid data type. Please choose either contact or phase.")
        exit()

    plt.ylabel(f"{short_leg_names[i]}", fontsize=12, rotation=90, labelpad=7)
    print(f"Leg {short_leg_names[i]} plotted.")

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

plt.subplots_adjust(left=0.05, right=0.98, hspace=0.7)
plt.suptitle(
    f"{args.data_type.capitalize()} Data from 6 Legs in the EigenBot",
    fontsize=12,
    y=0.95,
)

import os

directory = f"plotting"
if not os.path.exists(directory):
    os.makedirs(directory)
    print(f"Directory '{directory}' created.")

plt.savefig(f"{directory}/{args.data_name}_{args.comparison}_{args.data_type}.png")
