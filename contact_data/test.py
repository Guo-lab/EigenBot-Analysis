import matplotlib.pyplot as plt
import pandas as pd

# Example DataFrame
df = pd.DataFrame(
    {
        "Timestamp": list(range(20)),
        "Contact_1": [1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0],
    }
)

# Parameters
window_size = 10  # Adjust window size as needed


def moving_window_classification(df, window_size):
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
        majority_value = window_data["Contact_1"].mode().iloc[0]
        print(majority_value)
        classified.append(majority_value)

    return pd.Series(classified, index=df.index)


# Apply moving window classification
df["Classified"] = moving_window_classification(df, window_size)

# Plot
plt.figure(figsize=(10, 4))

plt.fill_between(
    df["Timestamp"],
    0,  # Bottom of the fill (y-axis start)
    1,  # Top of the fill (y-axis end)
    color="blue",  # Color for filling
    alpha=0.5,  # Transparency of fill
    where=df["Classified"] == 1,  # Condition for filling
)

plt.plot(df["Timestamp"], df["Contact_1"], "o", label="Contact State")

plt.xlabel("Timestamp")
plt.ylabel("Value")
plt.title("Filled Area Based on Moving Window Classification")
plt.legend()
plt.show()
