import re

log = """
===================================================================================================================
Calculating coupling strength for trans stance to swing
...................................................................................................................
contralateral coupling
......................
Sender: R3 -> Receiver: L3 

================================================
Calculating COUPLING STRENGTH...
================================================
Max:  0.6000, ∆t:  1.8000, Baseline:  0.3231
-----Coupling Strength:
-----Coupling Strength:
Rule 3:  0.2769
-----Coupling Efficacy:
Rule 3:  0.4090

Sender: L3 -> Receiver: R3 

================================================
Calculating COUPLING STRENGTH...
================================================
Max:  0.7273, ∆t:  0.2500, Baseline:  0.3179
-----Coupling Strength:
-----Coupling Strength:
Rule 3:  0.4094
-----Coupling Efficacy:
Rule 3:  0.6002

Sender: R2 -> Receiver: L2 

================================================
Calculating COUPLING STRENGTH...
================================================
Max:  0.7500, ∆t:  1.9000, Baseline:  0.4172
-----Coupling Strength:
-----Coupling Strength:
Rule 3:  0.3328
-----Coupling Efficacy:
Rule 3:  0.5711

Sender: L2 -> Receiver: R2 

================================================
Calculating COUPLING STRENGTH...
================================================
Max:  0.6923, ∆t:  0.2500, Baseline:  0.3848
-----Coupling Strength:
-----Coupling Strength:
Rule 3:  0.3075
-----Coupling Efficacy:
Rule 3:  0.4998

Sender: R1 -> Receiver: L1 

================================================
Calculating COUPLING STRENGTH...
================================================
Max:  0.9167, ∆t: -1.4000, Baseline:  0.7109
-----Coupling Strength:
-----Coupling Strength:
Rule 3:  0.2057
-----Coupling Efficacy:
Rule 3:  0.7117

Sender: L1 -> Receiver: R1 

================================================
Calculating COUPLING STRENGTH...
================================================
Max:  0.5000, ∆t: -1.7000, Baseline:  0.3972
-----Coupling Strength:
-----Coupling Strength:
Rule 3:  0.1028
-----Coupling Efficacy:
Rule 3:  0.1706

......................
ipsilateral coupling
......................
Sender: L1 -> Receiver: L2 

================================================
Calculating COUPLING STRENGTH...
================================================
Max:  0.8333, ∆t: -2.5000, Baseline:  0.4172
-----Coupling Strength:
-----Coupling Strength:
Rule 3:  0.4162
-----Coupling Efficacy:
Rule 3:  0.7140

Sender: L2 -> Receiver: L3 

================================================
Calculating COUPLING STRENGTH...
================================================
Max:  0.6154, ∆t: -1.0000, Baseline:  0.3231
-----Coupling Strength:
-----Coupling Strength:
Rule 3:  0.2922
-----Coupling Efficacy:
Rule 3:  0.4318

Sender: R1 -> Receiver: R2 

================================================
Calculating COUPLING STRENGTH...
================================================
Max:  0.7500, ∆t: -1.0000, Baseline:  0.3848
-----Coupling Strength:
-----Coupling Strength:
Rule 3:  0.3652
-----Coupling Efficacy:
Rule 3:  0.5936

Sender: R2 -> Receiver: R3 

================================================
Calculating COUPLING STRENGTH...
================================================
Max:  0.5833, ∆t: -1.9000, Baseline:  0.3179
-----Coupling Strength:
-----Coupling Strength:
Rule 3:  0.2655
-----Coupling Efficacy:
Rule 3:  0.3892

===================================================================================================================
Calculating coupling strength for trans swing to stance
...................................................................................................................
contralateral coupling
......................
Sender: R3 -> Receiver: L3 

================================================
Calculating COUPLING STRENGTH...
================================================
Min:  0.0000, ∆t: -2.5000, Baseline:  0.3231
Max:  0.8000, ∆t: -0.8000, Baseline:  0.3231
-----Coupling Strength:
-----Coupling Strength:
Rule 1: -0.3231
-----Coupling Strength:
Rule 2:  0.4769
-----Coupling Efficacy:
Rule 1:  1.0000
-----Coupling Efficacy:
Rule 2:  0.7045

Sender: L3 -> Receiver: R3 

================================================
Calculating COUPLING STRENGTH...
================================================
Min:  0.0909, ∆t: -0.1500, Baseline:  0.3179
Max:  0.7273, ∆t: -1.9500, Baseline:  0.3179
-----Coupling Strength:
-----Coupling Strength:
Rule 1: -0.2270
-----Coupling Strength:
Rule 2:  0.4094
-----Coupling Efficacy:
Rule 1:  0.7140
-----Coupling Efficacy:
Rule 2:  0.6002

Sender: R2 -> Receiver: L2 

================================================
Calculating COUPLING STRENGTH...
================================================
Min:  0.0833, ∆t:  1.8000, Baseline:  0.4172
Max:  0.7500, ∆t: -1.0500, Baseline:  0.4172
-----Coupling Strength:
-----Coupling Strength:
Rule 1: -0.3338
-----Coupling Strength:
Rule 2:  0.3328
-----Coupling Efficacy:
Rule 1:  0.8002
-----Coupling Efficacy:
Rule 2:  0.5711

Sender: L2 -> Receiver: R2 

================================================
Calculating COUPLING STRENGTH...
================================================
Min:  0.0769, ∆t: -0.2500, Baseline:  0.3848
Max:  0.6923, ∆t: -2.5000, Baseline:  0.3848
-----Coupling Strength:
-----Coupling Strength:
Rule 1: -0.3079
-----Coupling Strength:
Rule 2:  0.3075
-----Coupling Efficacy:
Rule 1:  0.8001
-----Coupling Efficacy:
Rule 2:  0.4998

Sender: R1 -> Receiver: L1 

================================================
Calculating COUPLING STRENGTH...
================================================
Min:  0.5000, ∆t: -1.1500, Baseline:  0.7109
Max:  0.8333, ∆t:  2.4000, Baseline:  0.7109
-----Coupling Strength:
-----Coupling Strength:
Rule 1: -0.2109
-----Coupling Strength:
Rule 2:  0.1224
-----Coupling Efficacy:
Rule 1:  0.2967
-----Coupling Efficacy:
Rule 2:  0.4234

Sender: L1 -> Receiver: R1 

================================================
Calculating COUPLING STRENGTH...
================================================
Min:  0.0000, ∆t: -2.5000, Baseline:  0.3972
Max:  0.8333, ∆t:  0.9000, Baseline:  0.3972
-----Coupling Strength:
-----Coupling Strength:
Rule 1: -0.3972
-----Coupling Strength:
Rule 2:  0.4362
-----Coupling Efficacy:
Rule 1:  1.0000
-----Coupling Efficacy:
Rule 2:  0.7235

......................
ipsilateral coupling
......................
Sender: L2 -> Receiver: L1 

================================================
Calculating COUPLING STRENGTH...
================================================
Min:  0.4615, ∆t: -1.8000, Baseline:  0.7109
Max:  0.8462, ∆t:  1.1500, Baseline:  0.7109
-----Coupling Strength:
-----Coupling Strength:
Rule 1: -0.2494
-----Coupling Strength:
Rule 2:  0.1352
-----Coupling Efficacy:
Rule 1:  0.3508
-----Coupling Efficacy:
Rule 2:  0.4678

Sender: L3 -> Receiver: L2 

================================================
Calculating COUPLING STRENGTH...
================================================
Min:  0.0000, ∆t: -1.5000, Baseline:  0.4172
Max:  0.8182, ∆t:  1.0000, Baseline:  0.4172
-----Coupling Strength:
-----Coupling Strength:
Rule 1: -0.4172
-----Coupling Strength:
Rule 2:  0.4010
-----Coupling Efficacy:
Rule 1:  1.0000
-----Coupling Efficacy:
Rule 2:  0.6880

Sender: R2 -> Receiver: R1 

================================================
Calculating COUPLING STRENGTH...
================================================
Min:  0.0000, ∆t: -2.5000, Baseline:  0.3972
Max:  0.8333, ∆t:  0.8000, Baseline:  0.3972
-----Coupling Strength:
-----Coupling Strength:
Rule 1: -0.3972
-----Coupling Strength:
Rule 2:  0.4362
-----Coupling Efficacy:
Rule 1:  1.0000
-----Coupling Efficacy:
Rule 2:  0.7235

Sender: R3 -> Receiver: R2 

================================================
Calculating COUPLING STRENGTH...
================================================
Min:  0.0000, ∆t: -2.5000, Baseline:  0.3848
Max:  0.9000, ∆t:  0.5000, Baseline:  0.3848
-----Coupling Strength:
-----Coupling Strength:
Rule 1: -0.3848
-----Coupling Strength:
Rule 2:  0.5152
-----Coupling Efficacy:
Rule 1:  1.0000
-----Coupling Efficacy:
Rule 2:  0.8374

"""


def get_strength_efficacy(log, strength_pattern, efficacy_pattern):
    strengths = re.findall(strength_pattern, log)
    efficacies = re.findall(efficacy_pattern, log)
    return [float(s) for s in strengths], [float(e) for e in efficacies]


strength_pattern = r"-----Coupling Strength:\nRule 3: +([0-9.]+)"
efficacy_pattern = r"-----Coupling Efficacy:\nRule 3: +([0-9.]+)"

strengths_rule3, efficacies_rule3 = get_strength_efficacy(
    log, strength_pattern, efficacy_pattern
)
print("Coupling Strengths:", strengths_rule3)
print("Coupling Efficacies:", efficacies_rule3)

strength_pattern = r"-----Coupling Strength:\nRule 2: +([0-9.]+)"
efficacy_pattern = r"-----Coupling Efficacy:\nRule 2: +([0-9.]+)"

strengths_rule2, efficacies_rule2 = get_strength_efficacy(
    log, strength_pattern, efficacy_pattern
)
print("Coupling Strengths:", strengths_rule2)
print("Coupling Efficacies:", efficacies_rule2)

strength_pattern = r"-----Coupling Strength:\nRule 1: (-?[0-9.]+)"
efficacy_pattern = r"-----Coupling Efficacy:\nRule 1: +([0-9.]+)"

strengths_rule1, efficacies_rule1 = get_strength_efficacy(
    log, strength_pattern, efficacy_pattern
)
print("Coupling Strengths:", strengths_rule1)
print("Coupling Efficacies:", efficacies_rule1)


val3 = 0.5
val2 = 0.3
val1 = 0.1


def get_formatted_val(val):
    if val < 0:
        return f"{val:.3f}"
    return f"{val:.4f}"


def get_percent_format(val):
    return f"{val * 100:.2f}%"


print()
print("Rule 1: \n\n")
print(
    f"""
                  {get_formatted_val(strengths_rule1[7])} / {get_percent_format(efficacies_rule1[7])}               {get_formatted_val(strengths_rule1[6])} / {get_percent_format(efficacies_rule1[6])}
        「L3」    ----------------------->   「L2」    ----------------------->   「L1」
        ||                            ||                            ||
         |                             |                             |
  {get_formatted_val(strengths_rule1[0])} | {get_formatted_val(strengths_rule1[1])}               {get_formatted_val(strengths_rule1[2])} | {get_formatted_val(strengths_rule1[3])}               {get_formatted_val(strengths_rule1[4])} | {get_formatted_val(strengths_rule1[5])}
  {get_percent_format(efficacies_rule1[0])} | {get_percent_format(efficacies_rule1[1])}               {get_percent_format(efficacies_rule1[2])} | {get_percent_format(efficacies_rule1[3])}               {get_percent_format(efficacies_rule1[4])} | {get_percent_format(efficacies_rule1[5])}
         |                             |                             |
         ||                            ||                            ||
        「R3」    ----------------------->   「R2」    ----------------------->   「R1」
                  {get_formatted_val(strengths_rule1[9])} / {get_percent_format(efficacies_rule1[9])}               {get_formatted_val(strengths_rule1[8])} / {get_percent_format(efficacies_rule1[8])}
"""
)
print()
print("Rule 2: \n")
print(
    f"""
                  {get_formatted_val(strengths_rule2[7])} / {get_percent_format(efficacies_rule2[7])}               {get_formatted_val(strengths_rule2[6])} / {get_percent_format(efficacies_rule2[6])}
        「L3」    ----------------------->   「L2」    ----------------------->   「L1」
        ||                            ||                            ||
         |                             |                             |
  {get_formatted_val(strengths_rule2[0])} | {get_formatted_val(strengths_rule2[1])}               {get_formatted_val(strengths_rule2[2])} | {get_formatted_val(strengths_rule2[3])}               {get_formatted_val(strengths_rule2[4])} | {get_formatted_val(strengths_rule2[5])}
  {get_percent_format(efficacies_rule2[0])} | {get_percent_format(efficacies_rule2[1])}               {get_percent_format(efficacies_rule2[2])} | {get_percent_format(efficacies_rule2[3])}               {get_percent_format(efficacies_rule2[4])} | {get_percent_format(efficacies_rule2[5])}
         |                             |                             |
         ||                            ||                            ||
        「R3」    ----------------------->   「R2」    ----------------------->   「R1」
                  {get_formatted_val(strengths_rule2[9])} / {get_percent_format(efficacies_rule2[9])}               {get_formatted_val(strengths_rule2[8])} / {get_percent_format(efficacies_rule2[8])}
"""
)
print()
print("Rule 3: \n")
print(
    f"""
                  {get_formatted_val(strengths_rule3[7])} / {get_percent_format(efficacies_rule3[7])}               {get_formatted_val(strengths_rule3[6])} / {get_percent_format(efficacies_rule3[6])}
        「L3」    <-----------------------   「L2」    <-----------------------   「L1」
        ||                            ||                            ||
         |                             |                             |
  {get_formatted_val(strengths_rule3[0])} | {get_formatted_val(strengths_rule3[1])}               {get_formatted_val(strengths_rule3[2])} | {get_formatted_val(strengths_rule3[3])}               {get_formatted_val(strengths_rule3[4])} | {get_formatted_val(strengths_rule3[5])}
  {get_percent_format(efficacies_rule3[0])} | {get_percent_format(efficacies_rule3[1])}               {get_percent_format(efficacies_rule3[2])} | {get_percent_format(efficacies_rule3[3])}               {get_percent_format(efficacies_rule3[4])} | {get_percent_format(efficacies_rule3[5])}
         |                             |                             |
         ||                            ||                            ||
        「R3」    <-----------------------   「R2」    <-----------------------   「R1」
                  {get_formatted_val(strengths_rule3[9])} / {get_percent_format(efficacies_rule3[9])}               {get_formatted_val(strengths_rule3[8])} / {get_percent_format(efficacies_rule3[8])}
"""
)
