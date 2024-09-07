# CASENAME=logging_phase_flat_extra7
CASENAME=logging_phase_flat_ablation_all1

echo "python format.py --file ${CASENAME}"
python format.py --file ${CASENAME}
. run.sh > ${CASENAME}.log
# 
# python coupling_plot.py > ${CASENAME}.plot.log