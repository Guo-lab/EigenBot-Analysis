import matplotlib.pyplot as plt
import numpy as np

# Parameters
total_time = 100  # seconds
dt = 0.05  # time step in seconds
num_legs = 6
time_points = int(total_time / dt)
time = np.arange(0, total_time, dt)

# Initialize gait pattern array
gait_pattern = np.zeros((time_points, num_legs))

# Define the cyclic pattern based on your input for each leg
# LF (Left Front), LM (Left Middle), LH (Left Hind), RF (Right Front), RM (Right Middle), RH (Right Hind)
import numpy as np


def generate_gait_pattern(swing_len, stance_len, total_repeats):
    """
    Generate a gait pattern of ones (swing phase) followed by zeros (stance phase).

    Parameters:
    - swing_len: Number of consecutive ones (swing phase).
    - stance_len: Number of consecutive zeros (stance phase).
    - total_repeats: Total number of times the swing/stance cycle should be repeated.
    - num_phases: Number of alternating phases (default 2).

    Returns:
    - pattern: Generated gait pattern.
    """
    # One complete cycle of the pattern (swing followed by stance)
    cycle = [1] * swing_len + [0] * stance_len

    # Repeat the cycle for the specified number of times
    pattern = cycle * total_repeats

    return pattern


# Parameters
k = 128  # Scaling factor (for example k zeros)
swing_len = k  # Number of consecutive ones (swing phase)
stance_len = k * 2  # Number of consecutive zeros (stance phase)
total_repeats = 2  # Total number of cycles (adjust as needed)

# Generate patterns for each leg
LF = generate_gait_pattern(swing_len, stance_len, total_repeats)
LM = np.roll(LF, shift=k * 4)  # Shift the pattern to mimic the timing for LM
LH = np.roll(LF, shift=k * 2)  # Shift the pattern to mimic the timing for LH
RF = np.roll(LF, shift=k * 4)  # Shift the pattern to mimic the timing for RF
RM = np.roll(LF, shift=k * 2)  # Shift the pattern to mimic the timing for RM
RH = LF  # RH has the same pattern as LF

# Print the generated patterns
print("LF:", LF)
print("LM:", LM)
print("LH:", LH)
print("RF:", RF)
print("RM:", RM)
print("RH:", RH)


# Repeat the pattern to cover the entire time period
pattern_length = len(LF)  # Length of the basic pattern cycle
for i in range(time_points):
    gait_pattern[i, 0] = LF[i % pattern_length]  # Left Front
    gait_pattern[i, 1] = LM[i % pattern_length]  # Left Middle
    gait_pattern[i, 2] = LH[i % pattern_length]  # Left Hind
    gait_pattern[i, 3] = RF[i % pattern_length]  # Right Front
    gait_pattern[i, 4] = RM[i % pattern_length]  # Right Middle
    gait_pattern[i, 5] = RH[i % pattern_length]  # Right Hind

# Plotting the gait pattern using fill_between to create blocky transitions
plt.figure(figsize=(12, 8))

# Loop through each leg and plot its stance/swing phases
for leg in range(num_legs):
    plt.fill_between(
        time,
        leg + 0.5,
        leg + 1.5,
        where=gait_pattern[:, leg] == 1,
        step="post",
        color="black",
        label=f"Leg {leg + 1}" if leg == 0 else "",
        alpha=1,
    )

# Add labels and title
plt.xlabel("Time (s)")
plt.ylabel("Leg number")
plt.title("Tetrapod Gait Pattern: Back-to-Front Wave")
# plt.grid(True)

# Add a custom legend for the black blocks representing swing phases
plt.legend(["Swing Phase"], loc="upper right")

# Display the plot
plt.savefig("Tetrapod Gait gait_pattern.png")
plt.show()


# LF[0, 1, 1, 0, 0, 0, 0, 1, 1, ...]
# LM[0, 0, 0, 0, 0, 1, 1, 0, 0, ...]
# LH[0, 0, 0, 1, 1, 0, 0, 0, 0, ...]
# RF[0, 0, 0, 0, 0, 1, 1, 0, 0, ...]
# RM[0, 0, 0, 1, 1, 0, 0, 0, 0, ...]
# RH[0, 1, 1, 0, 0, 0, 0, 1, 1, ...]
