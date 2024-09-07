NAME_LIST_1=(
    logging_phase_flat_1_0_5_contra_rule2
    logging_phase_flat_takes2_2_contra_rule2_0
    logging_phase_flat_extra7
    logging_phase_flat_extra6
    logging_phase_flat_extra5
)

NAME_LIST_1=(
    logging_phase_flat_2_0_5_contra_rule2
    logging_phase_flat_3_0_5_contra_rule2
    logging_phase_flat_4_0_5_contra_rule2
    logging_phase_flat_5_0_5_contra_rule2
    logging_phase_flat_6_0_5_contra_rule2
    # logging_phase_flat_extra1
    # logging_phase_flat_extra4
)


NAME_LIST_1=(
    logging_phase_terrain_1
    logging_phase_terrain_2_0_5_contra_rule2
    logging_phase_terrain_3_0_5_contra_rule2
    logging_phase_terrain_4_0_5_contra_rule2
    logging_phase_terrain_5_0_5_contra_rule2
)

NAME_LIST_1=(
    logging_phase_flat_left_hind_amputation1
    logging_phase_flat_left_hind_amputation2
    logging_phase_flat_left_hind_amputation3
    logging_phase_flat_left_hind_amputation4
    logging_phase_flat_left_hind_amputation5
    logging_phase_flat_left_hind_amputation6
)

NAME_LIST_1=( # Not for Double-Hill
    logging_phase_flat_1_middle_amputation_1
    logging_phase_flat_1_middle_amputation_2
    logging_phase_flat_1_middle_amputation_3
    logging_phase_flat_1_middle_amputation_4
    logging_phase_flat_1_middle_amputation_5
    logging_phase_flat_1_middle_amputation_6
)

# 2 both hind amputation
NAME_LIST_1=( # Not for Double-Hill
    logging_phase_flat_both_hind_amputation1
    logging_phase_flat_both_hind_amputation2
)

# 6 both middle amputation
NAME_LIST_1=( # Not for Double-Hill
    logging_phase_flat_both_middle_amputation1
    logging_phase_flat_both_middle_amputation2
    logging_phase_flat_both_middle_amputation3
    logging_phase_flat_both_middle_amputation4
    logging_phase_flat_both_middle_amputation5
    logging_phase_flat_both_middle_amputation6
)

NAME_LIST_1=(
    logging_phase_flat_ablation_all1
    logging_phase_flat_ablation_all2
    logging_phase_flat_ablation_all3
    logging_phase_flat_ablation_all4
)

# Initialize the base increment value
BASE_INCREMENT=500

# Loop through each file in NAME_LIST_1
for i in "${!NAME_LIST_1[@]}"; do
    FILE_NAME=${NAME_LIST_1[$i]}
    ADD_VALUE=$((BASE_INCREMENT * (i + 1)))

    # Call the Python script to update the Timestamp column
    python update_timestamps.py --file "${FILE_NAME}_data.csv" --add_value $ADD_VALUE
done