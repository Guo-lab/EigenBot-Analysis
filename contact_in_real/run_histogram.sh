# 5 Good Flat
NAME_LIST_1=(
    logging_phase_flat_1_0_5_contra_rule2
    logging_phase_flat_takes2_2_contra_rule2_0
    logging_phase_flat_extra7
    logging_phase_flat_extra6
    logging_phase_flat_extra5
)

# 7 Bad Flat
NAME_LIST_2=(
    logging_phase_flat_2_0_5_contra_rule2
    logging_phase_flat_3_0_5_contra_rule2
    logging_phase_flat_4_0_5_contra_rule2
    logging_phase_flat_5_0_5_contra_rule2
    logging_phase_flat_6_0_5_contra_rule2
    # logging_phase_flat_extra1
    # logging_phase_flat_extra4
)

# 5 Terrain
NAME_LIST_3=(
    logging_phase_terrain_1
    logging_phase_terrain_2_0_5_contra_rule2
    logging_phase_terrain_3_0_5_contra_rule2
    logging_phase_terrain_4_0_5_contra_rule2
    logging_phase_terrain_5_0_5_contra_rule2
)

# 6 amputation left_hind
NAME_LIST_4=(
    logging_phase_flat_left_hind_amputation1
    logging_phase_flat_left_hind_amputation2
    logging_phase_flat_left_hind_amputation3
    logging_phase_flat_left_hind_amputation4
    logging_phase_flat_left_hind_amputation5
    logging_phase_flat_left_hind_amputation6
)

# 6 one middle amputation
NAME_LIST_5=( # Not for Double-Hill
    logging_phase_flat_1_middle_amputation_1
    logging_phase_flat_1_middle_amputation_2
    logging_phase_flat_1_middle_amputation_3
    logging_phase_flat_1_middle_amputation_4
    logging_phase_flat_1_middle_amputation_5
    logging_phase_flat_1_middle_amputation_6
)

# 2 both hind amputation
NAME_LIST_6=( # Not for Double-Hill
    logging_phase_flat_both_hind_amputation1
    logging_phase_flat_both_hind_amputation2
    logging_phase_flat_both_hind_amputation3
    logging_phase_flat_both_hind_amputation5
    logging_phase_flat_both_hind_amputation6
)

# 6 both middle amputation
NAME_LIST_7=( # Not for Double-Hill
    logging_phase_flat_both_middle_amputation1
    logging_phase_flat_both_middle_amputation2
    logging_phase_flat_both_middle_amputation3
    logging_phase_flat_both_middle_amputation4
    logging_phase_flat_both_middle_amputation5
    logging_phase_flat_both_middle_amputation6
)

NAME_LIST_8=( # ablation study
    logging_phase_flat_ablation_all1
    logging_phase_flat_ablation_all2
    logging_phase_flat_ablation_all3
    logging_phase_flat_ablation_all4
)

LEG_NUMBERS=(2 5 13 17 18 19)

# GROUP_NAMES=("NAME_LIST_1" "NAME_LIST_2" "NAME_LIST_3" "NAME_LIST_4" "NAME_LIST_5" "NAME_LIST_6" "NAME_LIST_7")
GROUP_NAME="NAME_LIST_7"

eval NAME_LIST=\${$GROUP_NAME[@]}
NAME_LIST_LENGTH=${#NAME_LIST[@]}
SLEEP_TIME=$((NAME_LIST_LENGTH * 2))

echo "NAME_LIST: ${NAME_LIST}, with length: ${NAME_LIST_LENGTH}"

for LEG_NUMBER in ${LEG_NUMBERS[@]}; do
    for NAME in ${NAME_LIST[@]}; do
        echo "python entropy.py --file ${NAME} --leg_number ${LEG_NUMBER} --group $GROUP_NAME"
        python entropy.py --file ${NAME} --leg_number ${LEG_NUMBER} --group $GROUP_NAME
    done

    sleep $SLEEP_TIME
    echo "python histogram_plot.py --leg_number ${LEG_NUMBER} --group $GROUP_NAME"
    python histogram_plot.py --leg_number ${LEG_NUMBER} --group $GROUP_NAME
    sleep 3
done
