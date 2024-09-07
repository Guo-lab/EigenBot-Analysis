import argparse

import matplotlib.pyplot as plt
import pandas as pd

parser = argparse.ArgumentParser(description="as the file label")

parser.add_argument(
    "--txt_file_name",
    type=str,
    default="",
    help="file to parse",
)

args = parser.parse_args()
print(args)


############################################
# ========= Parsing the arguments ======== #
############################################
data_name = args.txt_file_name
with open(data_name + ".txt", "r") as file:
    data = file.readlines()

ros_timestamp = [line.split(":")[0].split(",")[0] for line in data]
for line in data:
    if len(line.split(":")[0].split(",")) == 1:
        data.remove(line)

filtered_data_lines = [line.split(":")[0].split(",")[1] for line in data]
print(f"Found {len(filtered_data_lines)} lines of data.")

processed_data_cnt = 0
timestamps = []
phases = []
contacts = []
legs = {}
for each_line in filtered_data_lines:
    filtered_line = each_line.split("Z")[1].split(" ")
    if len(filtered_line) != 4:
        continue
    processed_data_cnt += 1

    module_number = int(filtered_line[0])
    phase = int(filtered_line[1])
    print(filtered_line)
    contact = int(filtered_line[2])

    timestamp = float(filtered_line[3])

    assert isinstance(module_number, int), "module number should be an integer"
    assert isinstance(phase, int) and phase in range(4), "phase be an integer 0-3"
    assert isinstance(contact, int) and contact in range(2), "contact be (0/1)"
    assert isinstance(timestamp, float), "timestamp should be a float"
    phase = 1 if phase >= 2 else 0  # 0: swing (0,1), 1: stance (2,3)

    if module_number not in legs:
        legs[module_number] = {"phase": [], "contact": [], "timestamp": []}

    legs[module_number]["phase"].append(phase)
    legs[module_number]["contact"].append(contact)
    legs[module_number]["timestamp"].append(timestamp)

print(f"Processed {processed_data_cnt} lines of data.")
print(f"Found {len(legs)} legs in the data.")
print()

min_timestamp = float("inf")
max_timestamp = -float("inf")
for each_leg in legs:
    leg_max_timestamp = max(legs[each_leg]["timestamp"])
    leg_min_timestamp = min(legs[each_leg]["timestamp"])
    print(f"Leg {each_leg} has {leg_min_timestamp} - {leg_max_timestamp}.")

    if leg_max_timestamp < min_timestamp:
        min_timestamp = leg_max_timestamp
    if leg_min_timestamp > max_timestamp:
        max_timestamp = leg_min_timestamp

print(f"Timestamp range: {max_timestamp} - {min_timestamp}")
print()

combined_df = pd.DataFrame()
plotting_dict = {}
for each_leg in legs:
    assert len(legs[each_leg]["phase"]) == len(legs[each_leg]["contact"])
    assert len(legs[each_leg]["phase"]) == len(legs[each_leg]["timestamp"])

    df = pd.DataFrame(
        {
            "Leg": each_leg,
            "Timestamp": legs[each_leg]["timestamp"],
            "Phase": legs[each_leg]["phase"],
            "Contact": legs[each_leg]["contact"],
        }
    )
    df = df.sort_values(by="Timestamp")
    df = df[(df["Timestamp"] >= max_timestamp) & (df["Timestamp"] <= min_timestamp)]

    plotting_dict[each_leg] = df
    print(f"Leg {each_leg} has {len(df)} data points.")

    if each_leg == 5:
        filtered_df = df[df["Phase"] == 1]
        print(f"Leg {each_leg} has {len(filtered_df)} stance data points.")

    combined_df = pd.concat([combined_df, df], ignore_index=True)

print()
combined_df.to_csv(f"{data_name}_data.csv", index=False)
