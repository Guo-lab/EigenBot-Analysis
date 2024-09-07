echo "==================================================================================================================="
echo "Calculating coupling strength for trans stance to swing"
echo "..................................................................................................................."
echo "contralateral coupling"
echo "......................"
python coupling_strength.py --trans trans01 --which2which R3:L3 --leg_pairs contra
python coupling_strength.py --trans trans01 --which2which L3:R3 --leg_pairs contra
python coupling_strength.py --trans trans01 --which2which R2:L2 --leg_pairs contra
python coupling_strength.py --trans trans01 --which2which L2:R2 --leg_pairs contra
python coupling_strength.py --trans trans01 --which2which R1:L1 --leg_pairs contra
python coupling_strength.py --trans trans01 --which2which L1:R1 --leg_pairs contra
echo "......................"
echo "ipsilateral coupling"
echo "......................"
python coupling_strength.py --trans trans01 --which2which L1:L2 --leg_pairs ipsi
python coupling_strength.py --trans trans01 --which2which L2:L3 --leg_pairs ipsi
python coupling_strength.py --trans trans01 --which2which R1:R2 --leg_pairs ipsi
python coupling_strength.py --trans trans01 --which2which R2:R3 --leg_pairs ipsi

echo "==================================================================================================================="
echo "Calculating coupling strength for trans swing to stance"
echo "..................................................................................................................."
echo "contralateral coupling"
echo "......................"
python coupling_strength.py --trans trans10 --which2which R3:L3 --leg_pairs contra
python coupling_strength.py --trans trans10 --which2which L3:R3 --leg_pairs contra
python coupling_strength.py --trans trans10 --which2which R2:L2 --leg_pairs contra
python coupling_strength.py --trans trans10 --which2which L2:R2 --leg_pairs contra
python coupling_strength.py --trans trans10 --which2which R1:L1 --leg_pairs contra
python coupling_strength.py --trans trans10 --which2which L1:R1 --leg_pairs contra
echo "......................"
echo "ipsilateral coupling"
echo "......................"
python coupling_strength.py --trans trans10 --which2which L2:L1 --leg_pairs ipsi
python coupling_strength.py --trans trans10 --which2which L3:L2 --leg_pairs ipsi
python coupling_strength.py --trans trans10 --which2which R2:R1 --leg_pairs ipsi
python coupling_strength.py --trans trans10 --which2which R3:R2 --leg_pairs ipsi