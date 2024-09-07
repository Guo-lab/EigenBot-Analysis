#############################################################
########## neural flat + ipsi/contra contact/phase ##########
#############################################################
python3 contact_plot.py \
    --terrain flat --control neural \
    --curve_method one \
    --comparison ipsi \
    --data_type contact

python3 contact_plot.py \
    --terrain flat --control neural \
    --curve_method one \
    --comparison contra \
    --data_type contact

python3 contact_plot.py \
    --terrain flat --control neural \
    --curve_method one \
    --comparison ipsi \
    --data_type phase

python3 contact_plot.py \
    --terrain flat --control neural \
    --curve_method one \
    --comparison contra \
    --data_type phase



#############################################################
###### neural hill terrain + ipsi/contra contact/phase ######
#############################################################
python3 contact_plot.py \
    --terrain hill --control neural \
    --curve_method one \
    --comparison ipsi \
    --data_type contact

python3 contact_plot.py \
    --terrain hill --control neural \
    --curve_method one \
    --comparison contra \
    --data_type contact

python3 contact_plot.py \
    --terrain hill --control neural \
    --curve_method one \
    --comparison ipsi \
    --data_type phase

python3 contact_plot.py \
    --terrain hill --control neural \
    --curve_method one \
    --comparison contra \
    --data_type phase

#############################################################
###### predefined flat + ipsi/contra contact/phase ##########
#############################################################
python3 contact_plot.py \
    --terrain flat --control predefined \
    --curve_method one \
    --comparison ipsi \
    --data_type contact

python3 contact_plot.py \
    --terrain flat --control predefined \
    --curve_method one \
    --comparison contra \
    --data_type contact

python3 contact_plot.py \
    --terrain flat --control predefined \
    --curve_method one \
    --comparison ipsi \
    --data_type phase

python3 contact_plot.py \
    --terrain flat --control predefined \
    --curve_method one \
    --comparison contra \
    --data_type phase



#############################################################
##### predefined hill terrain + ipsi/contra contact/phase ###
#############################################################
python3 contact_plot.py \
    --terrain hill --control predefined \
    --curve_method one \
    --comparison ipsi \
    --data_type contact

python3 contact_plot.py \
    --terrain hill --control predefined \
    --curve_method one \
    --comparison contra \
    --data_type contact

python3 contact_plot.py \
    --terrain hill --control predefined \
    --curve_method one \
    --comparison ipsi \
    --data_type phase

python3 contact_plot.py \
    --terrain hill --control predefined \
    --curve_method one \
    --comparison contra \
    --data_type phase


# ===========================================================
#                    Curve Walking
# ===========================================================

#############################################################
####### curve one neural-flat + ipsi/contra contact/phase ##########
#############################################################
python3 contact_plot.py \
    --terrain flat --control neural \
    --curve True --curve_method one \
    --comparison ipsi \
    --data_type contact

python3 contact_plot.py \
    --terrain flat --control neural \
    --curve True --curve_method one \
    --comparison contra \
    --data_type contact

python3 contact_plot.py \
    --terrain flat --control neural \
    --curve True --curve_method one \
    --comparison ipsi \
    --data_type phase

python3 contact_plot.py \
    --terrain flat --control neural \
    --curve True --curve_method one \
    --comparison contra \
    --data_type phase



#############################################################
##### curve two neural-flat + ipsi/contra contact/phase #####
#############################################################
python3 contact_plot.py \
    --terrain flat --control neural \
    --curve True --curve_method two \
    --comparison ipsi \
    --data_type contact

python3 contact_plot.py \
    --terrain flat --control neural \
    --curve True --curve_method two \
    --comparison contra \
    --data_type contact

python3 contact_plot.py \
    --terrain flat --control neural \
    --curve True --curve_method two \
    --comparison ipsi \
    --data_type phase

python3 contact_plot.py \
    --terrain flat --control neural \
    --curve True --curve_method two \
    --comparison contra \
    --data_type phase

