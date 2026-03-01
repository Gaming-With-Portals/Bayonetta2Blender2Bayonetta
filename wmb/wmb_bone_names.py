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
    11: "wrist_R",
    12: "hand_R",
    13: "collar_L",
    14: "shoulder_L",
    15: "upper_arm_L",
    16: "lower_arm_L",
    17: "wrist_L",
    18: "hand_L",
    19: "pelvis",
    20: "upper_leg_R",
    21: "lower_leg_R",
    22: "foot_R",
    23: "toe_R",
    24: "upper_leg_L",
    25: "lower_leg_L",
    26: "foot_L",
    27: "toe_L"
}

def getBoneName(glob_id, loc_id, override=False):
    if (glob_id in wmb0_bonenames and override):
        return wmb0_bonenames[glob_id]
    else:
        return f"bone{loc_id:04}"