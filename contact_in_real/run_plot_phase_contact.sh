#logging_phase_terrain_2_0_5_contra_rule2 #logging_phase_terrain_1

# Good flat ones
NAME=logging_phase_flat_1_0_5_contra_rule2
NAME=logging_phase_flat_takes2_2_contra_rule2_0
NAME=logging_phase_flat_extra7

NAME=logging_phase_flat_both_hind_amputation6


echo 'python3 parsing_data.py'
python3 parsing_data.py \
    --txt_file_name ${NAME}

echo 'python3 plot_data.py'

python3 plot_data.py \
    --terrain flat --control neural \
    --comparison ipsi \
    --data_type contact \
    --data_name ${NAME}

python3 plot_data.py \
    --terrain flat --control neural \
    --comparison ipsi \
    --data_type phase \
    --data_name ${NAME}