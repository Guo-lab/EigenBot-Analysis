GROUP_1=(
    logging_phase_flat_1_0_5_contra_rule2 
    logging_phase_flat_2_0_5_contra_rule2 
    logging_phase_flat_3_0_5_contra_rule2 
    logging_phase_flat_4_0_5_contra_rule2 
    logging_phase_flat_5_0_5_contra_rule2 
    logging_phase_flat_6_0_5_contra_rule2
)

GROUP_2=(
    logging_phase_terrain_1
    logging_phase_terrain_2_0_5_contra_rule2
    logging_phase_terrain_3_0_5_contra_rule2
    logging_phase_terrain_4_0_5_contra_rule2
    logging_phase_terrain_5_0_5_contra_rule2
)


# 6 amputation left_hind
GROUP_3=(
    logging_phase_flat_left_hind_amputation1
    logging_phase_flat_left_hind_amputation2
    logging_phase_flat_left_hind_amputation3
    logging_phase_flat_left_hind_amputation4
    logging_phase_flat_left_hind_amputation5
    logging_phase_flat_left_hind_amputation6
)

# 6 one middle amputation
GROUP_4=( # Not for Double-Hill
    logging_phase_flat_1_middle_amputation_1
    logging_phase_flat_1_middle_amputation_2
    logging_phase_flat_1_middle_amputation_3
    logging_phase_flat_1_middle_amputation_4
    logging_phase_flat_1_middle_amputation_5
    logging_phase_flat_1_middle_amputation_6
)

# 2 both hind amputation
GROUP_5=( # Not for Double-Hill
    logging_phase_flat_both_hind_amputation1
    logging_phase_flat_both_hind_amputation2
    logging_phase_flat_both_hind_amputation3
    logging_phase_flat_both_hind_amputation5
    logging_phase_flat_both_hind_amputation6
)

# 6 both middle amputation
GROUP_6=( # Not for Double-Hill
    logging_phase_flat_both_middle_amputation1
    logging_phase_flat_both_middle_amputation2
    logging_phase_flat_both_middle_amputation3
    logging_phase_flat_both_middle_amputation4
    logging_phase_flat_both_middle_amputation5
    logging_phase_flat_both_middle_amputation6
)


GROUP_7=( # ablation study
    logging_phase_flat_ablation_all1
    logging_phase_flat_ablation_all2
    logging_phase_flat_ablation_all3
    logging_phase_flat_ablation_all4
)


python all_hills_radius.py  \
    --group "${GROUP_1[@]}" --tag flat


python all_hills_radius.py  \
    --group "${GROUP_2[@]}" --tag terrain


python all_hills_radius.py  \
    --group "${GROUP_3[@]}" --tag amputation_l3


python all_hills_radius.py  \
    --group "${GROUP_4[@]}" --tag amputation_r2

python all_hills_radius.py  \
    --group "${GROUP_5[@]}" --tag amputation_r3l3

python all_hills_radius.py  \
    --group "${GROUP_6[@]}" --tag amputation_r2l2

python all_hills_radius.py  \
    --group "${GROUP_7[@]}" --tag ablation

