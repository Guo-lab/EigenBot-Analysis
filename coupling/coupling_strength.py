import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

verbose = False

parser = argparse.ArgumentParser(description="as the file label")

parser.add_argument(
    "--trans",
    type=str,
    default="",
    help="which transition calculating on: trans10 (swing to stance) or trans01 (stance to swing)",
)
parser.add_argument(
    "--which2which",
    type=str,
    default="",
    help="which leg is the sender and which is the receiver, e.g. L3:L2, Left Hind to Left Middle",
)
parser.add_argument(
    "--window_size",
    type=float,
    default=5,  # 4.5
    help="the width of the window to calculate the coupling strength",
)
parser.add_argument(
    "--dt",
    type=float,
    default=0.05,
    help="time step of the data",
)
parser.add_argument(
    "--verbose",
    type=bool,
    default=False,
    help="whether to print out the results",
)
parser.add_argument(
    "--leg_pairs",
    type=str,
    default="",
    help="ipsi or contra",
)

args = parser.parse_args()
if verbose:
    print(args)


######### CSV file reading #########
csv_file_name = "interpolated_data.csv"  # "phase_data_flat_neural.csv"
raw_df = pd.read_csv(csv_file_name)
if verbose:
    print(raw_df, "\n")


def moving_window_classification(df, leg_index, window_size=2):
    """
    Classify each timestamp as 0 or 1 based on the majority value within a window around each timestamp.

    Parameters:
    - df: DataFrame with 'Timestamp' and 'Contact_1' columns.
    - window_size: The size of the window for classification.

    Returns:
    - A Series with the classified values.
    """
    classified = []
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


# Filter the raw phase data.
for leg_index in range(6):
    raw_df[f"Contact_{leg_index + 1}"] = moving_window_classification(
        raw_df, leg_index, window_size=2
    )

# Convert Swing from 0 to 1 and Stance from 1 to 0.
contact_columns_ = [col for col in raw_df.columns if col.startswith("Contact")]
df = raw_df.copy()
df[contact_columns_] = df[contact_columns_].apply(lambda x: 1 - x)
if verbose:
    print(df, "\n")
    print(df.index, "\n")

win_l = args.window_size / 2 * -1
win_r = args.window_size / 2
if verbose:
    print(f"Calculating coupling strength with window from {win_l} - {win_r}.", "\n")

sender = args.which2which.split(":")[0]
receiver = args.which2which.split(":")[1]
mapping_which2column = {
    "L1": "Contact_1",
    "L2": "Contact_2",
    "L3": "Contact_3",
    "R1": "Contact_4",
    "R2": "Contact_5",
    "R3": "Contact_6",
}
print(f"Sender: {sender} -> Receiver: {receiver}", "\n")

sender_event = "touch_down" if args.trans == "trans10" else "lift_off"


# Get the sender events (timestamp list) from the DataFrame.
def getting_sender_events(df, sender_column, sender_events):
    """
    Get the sender events from the DataFrame.
        The sender events are the touch-down or lift-off moments of the sender leg.
    """
    df["Previous"] = df[sender_column].shift(1)
    df = df.dropna(subset="Previous")

    if sender_events == "touch_down":
        touch_down_moments = df[(df["Previous"] == 1) & (df[sender_column] == 0)]
        time_list = touch_down_moments["Timestamp"].tolist()
        if verbose:
            print("touch down moment: ", time_list, "\n")
        return touch_down_moments

    elif sender_events == "lift_off":
        lift_off_moments = df[(df["Previous"] == 0) & (df[sender_column] == 1)]
        time_list = lift_off_moments["Timestamp"].tolist()
        if verbose:
            print("Lift off moments: ", time_list, "\n")
        return lift_off_moments

    else:
        print("Invalid sender event.")
        return None


sender_event_df = getting_sender_events(df, mapping_which2column[sender], sender_event)
if verbose:
    print(f"{sender_event} DataFrame: ", sender_event_df, "\n")
    print(f"{sender_event} length: ", sender_event_df["Timestamp"].size, "\n")


# Calculate Avg Swing Probability of the receiver
avg_swing_probability_baseline = df[mapping_which2column[receiver]].mean()


def calculate_probability(df, sender_moments, receiver_column, window_size, step_size):
    """
    Calculate the probability of swing mode (Contact is 1) for the receiver leg around each touch-down (lift-off) moment of the sender leg.
        Define the window around each touch-down (lift-up) moment
        Filter the DataFrame within the time window
        Calculate the probability of swing mode (Contact is 1)
        Then, calculate the average probability across all touch-down moments

    Parameters:
        df: DataFrame with 'Timestamp' and 'Contact_1' columns.
        sender_moments: DataFrame with 'Timestamp' column.
        receiver_column: The column name of the receiver leg.
        window_size: The size of the window for classification.
        step_size: The step size for moving the window.

    Returns:
        The average probability of swing mode for the receiver leg around each touch-down (lift-off) moment of the sender leg.
    """
    print(
        "================================================\nCalculating COUPLING STRENGTH...\n================================================",
    )
    if verbose:
        print(f"Window size: {window_size};\n Step size: {step_size}", "\n")
    window_size = window_size / step_size
    if verbose:
        print(f"x axis (actual) length: ", window_size, "\n")
        print(f"Receiver column: {receiver_column}", "\n")
        print(f"Sender moments: {sender_moments}", "\n")

    accumulated_values = np.zeros(int(window_size) + 1)
    window_values = np.zeros(int(window_size) + 1)
    if verbose:
        print("Window values: ", window_values.shape, "\n")
        print("Window values: ", window_values, "\n")

    for t in sender_moments.index:
        # print(f"Touch down moment Index: {t}")
        window_start = t - (window_size / 2)
        window_end = t + (window_size / 2)
        # print(f"Window start: {window_start}; Window end: {window_end}", "\n")

        window_df = df[(df.index >= window_start) & (df.index <= window_end)]
        # print("Window DataFrame size: ", window_df.shape, "\n")
        # print("Window DataFrame: ", window_df, "\n")
        if window_values.shape[0] != window_df[receiver_column].values.shape[0]:
            if verbose:
                print(
                    "Window values shape is not equal to window_df values shape. Skip this window."
                )
            continue
        window_values[: int(window_size) + 1] = window_df[receiver_column].values

        if not window_df.empty:
            accumulated_values += window_values
            # print(f"For the window, the accumulated values are: {accumulated_values}\n")
        else:
            print("Window DataFrame is empty.")
            return None

    avg_probability = accumulated_values / sender_moments["Timestamp"].size
    return avg_probability


# Calculate coupling strength
avg_swing_probability = calculate_probability(
    df, sender_event_df, mapping_which2column[receiver], args.window_size, args.dt
)
if verbose:
    print(
        f"Average swing probability: {avg_swing_probability}\nwith size: {avg_swing_probability.shape}\n"
    )
assert avg_swing_probability is not None, "The average swing probability is None."


# Plotting
plt.figure(figsize=(12, 6))
time_range = np.linspace(win_l, win_r, int(args.window_size / args.dt) + 1)
plt.plot(time_range, avg_swing_probability, label="Swing Probability", color="blue")

# Add baseline
plt.axhline(
    y=avg_swing_probability_baseline, color="red", linestyle="--", label="Baseline"
)

# Set axis labels and title
plt.xlabel("Time (s)")
plt.ylabel("Average Swing Probability")
plt.title(f"Likelihood of Protraction in Anterior Receiver Leg {sender} -> {receiver}")

# Set x-limits to match the window size
plt.xlim(-2.25, 2.25)

# Add legend
plt.legend()

# Display the plot
# plt.grid(True)
directory = "coupling_strength_plots"
plt.savefig(f"{directory}/coupling_strength_{args.trans}_{sender}_{receiver}.png")
if verbose:
    plt.show()

max_swing_probability = np.max(avg_swing_probability)
min_swing_probability = np.min(avg_swing_probability)
min_indices = np.where(avg_swing_probability == min_swing_probability)[0]
max_indices = np.where(avg_swing_probability == max_swing_probability)[0]


def closest_to_moment(indices):
    return indices[np.argmin(np.abs(indices))]


if args.trans == "trans10":
    print(
        f"Min: {min_swing_probability: .4f}, ∆t: {time_range[closest_to_moment(min_indices)]: .4f}, Baseline: {avg_swing_probability_baseline: .4f}"
    )
print(
    f"Max: {max_swing_probability: .4f}, ∆t: {time_range[closest_to_moment(max_indices)]: .4f}, Baseline: {avg_swing_probability_baseline: .4f}"
)

print("-----Coupling Strength:")
if args.leg_pairs == "ipsi":
    if args.trans == "trans10":
        print("-----Coupling Strength:")
        print(f"Rule 1: {min_swing_probability - avg_swing_probability_baseline: .4f}")
        print("-----Coupling Strength:")
        print(f"Rule 2: {max_swing_probability - avg_swing_probability_baseline: .4f}")
        print("-----Coupling Efficacy:")
        print(
            f"Rule 1: {(avg_swing_probability_baseline - min_swing_probability) / avg_swing_probability_baseline: .4f}"
        )
        print("-----Coupling Efficacy:")
        print(
            f"Rule 2: {(max_swing_probability - avg_swing_probability_baseline) / (1 - avg_swing_probability_baseline): .4f}"
        )
    elif args.trans == "trans01":
        print("-----Coupling Strength:")
        print(f"Rule 3: {max_swing_probability - avg_swing_probability_baseline: .4f}")
        print("-----Coupling Efficacy:")
        print(
            f"Rule 3: {(max_swing_probability - avg_swing_probability_baseline) / (1 - avg_swing_probability_baseline): .4f}"
        )
    else:
        print("Invalid transition type.")
        exit(1)

elif args.leg_pairs == "contra":
    if args.trans == "trans10":
        print("-----Coupling Strength:")
        print(f"Rule 1: {min_swing_probability - avg_swing_probability_baseline: .4f}")
        print("-----Coupling Strength:")
        print(f"Rule 2: {max_swing_probability - avg_swing_probability_baseline: .4f}")
        print("-----Coupling Efficacy:")
        print(
            f"Rule 1: {(avg_swing_probability_baseline - min_swing_probability) / avg_swing_probability_baseline: .4f}"
        )
        print("-----Coupling Efficacy:")
        print(
            f"Rule 2: {(max_swing_probability - avg_swing_probability_baseline) / (1 - avg_swing_probability_baseline): .4f}"
        )
    elif args.trans == "trans01":
        print("-----Coupling Strength:")
        print(f"Rule 3: {max_swing_probability - avg_swing_probability_baseline: .4f}")
        print("-----Coupling Efficacy:")
        print(
            f"Rule 3: {(max_swing_probability - avg_swing_probability_baseline) / (1 - avg_swing_probability_baseline): .4f}"
        )

else:
    print("Invalid leg pairs.")
    exit(1)

print()
