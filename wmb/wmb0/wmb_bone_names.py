wmb0_bonenames = {
    0: "ROOT",
    1: "HIP",
    2: "spine_1",
    3: "spine_2",
    4: "spine_3",
    5: "neck",
    6: "head",
    7: "collar_R",
    8: "shoulder_R",
    9: "upper_arm_R",
    10: "lower_arm_R",
    29: "elbow_R",    
    30: "wrist_R",
    12: "hand_R",
    13: "collar_L",
    14: "shoulder_L",
    15: "upper_arm_L",
    16: "lower_arm_L",
    31: "elbow_L",   
    32: "wrist_L",
    18: "hand_L",
    28: "breasts",
    19: "pelvis",
    20: "upper_leg_R",
    21: "lower_leg_R",
    22: "foot_R",
    23: "toe_R",
    34: "knee_R",
    33: "thigh_R",
    24: "upper_leg_L",
    25: "lower_leg_L",
    26: "foot_L",
    27: "toe_L",
    36: "knee_L",
    35: "thigh_L",    
}

wmb0_b2_bonenames = {
    0: "ROOT",
    1: "HIP",
    2: "spine_1",
    3: "spine_2",
    4: "spine_3",
    5: "neck",
    6: "head",
    11: "collar_R",
    12: "shoulder_R",
    13: "upper_arm_R",
    14: "lower_arm_R",
    15: "elbow_R",
    16: "wrist_R",
    17: "hand_R",
    38: "collar_L",
    39: "shoulder_L",
    40: "upper_arm_L",
    41: "lower_arm_L",
    42: "elbow_L",
    43: "wrist_L",
    44: "hand_L",
    65: "breasts",
    68: "pelvis",
    69: "upper_leg_R",
    70: "lower_leg_R",
    71: "foot_R",
    72: "toe_R",
    73: "knee_R",
    79: "thigh_R",
    74: "upper_leg_L",
    75: "lower_leg_L",
    76: "foot_L",
    77: "toe_L",
    78: "knee_L",
    80: "thigh_L",
}

def getBoneName(glob_id, loc_id, override=False):
    if (glob_id in wmb0_bonenames and override):
        return wmb0_bonenames[glob_id]
    else:
        return f"bone{loc_id:04}"
    
def getBoneNameB2(glob_id, loc_id, override=False):
    if (glob_id in wmb0_b2_bonenames and override):
        return wmb0_b2_bonenames[glob_id]
    else:
        return f"bone{loc_id:04}"