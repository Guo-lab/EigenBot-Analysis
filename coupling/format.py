import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser(description="Update 'Timestamp' column in a CSV file.")
parser.add_argument("--file", type=str, required=True, help="Path to the CSV file.")

args = parser.parse_args()

df = pd.read_csv(f"{args.file}_data.csv")
new_timestamps = np.arange(0, len(df) // len(df["Leg"].unique()) * 0.05, 0.05)

pivoted_df = df.pivot(index="Timestamp", columns="Leg", values="Phase")
pivoted_df.iloc[0] = pivoted_df.iloc[0].fillna(0)
# print(pivoted_df)
# print()
pivoted_df = pivoted_df.fillna(method="ffill")
# print(pivoted_df)

df = pivoted_df.copy()

start = int(np.floor(df.index.min()))
end = int(np.ceil(df.index.max()))

new_index = np.arange(start, end + 0.05, 0.05)
# print(new_index)
# print(len(new_index))
df_reindexed = df.reindex(new_index, method="nearest")
# print(df_reindexed)
df_interpolated = df_reindexed.interpolate(method="linear")
df_interpolated = df_interpolated.reset_index()
# print(df_interpolated)

new_order = ["Timestamp", 5, 2, 18, 13, 19, 17]
df_interpolated = df_interpolated[new_order]
# print("reordered one: ", df_interpolated)

# Step 2: Rename the columns from leg numbers to 'Contact_1', 'Contact_2', etc.
new_column_names = [
    "Timestamp",
    "Contact_1",
    "Contact_2",
    "Contact_3",
    "Contact_4",
    "Contact_5",
    "Contact_6",
]

df_interpolated.columns = new_column_names
# Step 3: Reindex the DataFrame with a new index (0, 1, 2, ...)
df_final = df_interpolated.reset_index(drop=True)

# Display the final DataFrame
print(df_final)
df_final.to_csv("interpolated_data.csv", index=False)


# # Plotting again
# plt.figure(figsize=(12, 8))

# mapping = {0: 2, 1: 5, 2: 13, 3: 17, 4: 18, 5: 19}
# # Loop through each leg and plot its stance/swing phases
# for leg in range(6):
#     plt.fill_between(
#         df_final["Timestamp"],
#         leg + 0.5,
#         leg + 1.5,
#         where=df_final[f"Contact_{leg+1}"] == 0,
#         step="post",
#         color="black",
#         label=f"Leg {leg + 1}" if leg == 0 else "",
#         alpha=1,
#     )

# # Add labels and title
# plt.xlabel("Time (s)")
# plt.ylabel("Leg number")
# plt.title("Tetrapod Gait Pattern: Back-to-Front Wave")
# # plt.grid(True)

# # Add a custom legend for the black blocks representing swing phases
# plt.legend(["Stance Phase"], loc="upper right")
# plt.savefig("Tetrapod Gait gait_pattern.png")

######################
# Calculate the entropy of the gaits
# from scipy.stats import entropy


# def calculate_entropy(time_series):
#     data = np.array(time_series)
#     probs = [np.mean(data == 0), np.mean(data == 1)]
#     return entropy(probs, base=2)


# entropy_values = {}
# for column in df_final.columns:
#     if column.startswith("Contact_"):
#         print(column)
#         series = df_final[column]
#         entropy_value = calculate_entropy(series)
#         entropy_values[column] = entropy_value

# print("Entropy values:", entropy_values)
