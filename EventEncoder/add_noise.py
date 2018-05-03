import numpy as np
import progressbar
import os
import re
import glob
import random
from utils import load_samples, save_samples

kDataRoot = "C:/Users/JM/Desktop/Data/ETRIrelated/BMVC"
kActionRoot = os.path.join(kDataRoot, "etri_action_data_neck_point_0_0")
kActionLen = 30
kPoseBaseIdx = 18
nosing_rate = 0.1
nNoisePoint = 3


def generation_dirty_samples(_poselet, ):

    noise_poselet = _poselet
    noise_index = random.shuffle(list(range(0, kPoseBaseIdx * kActionLen)))

    for idx in noise_index:
        _poselet[2 * idx + 0] = 0
        _poselet[2 * idx + 1] = 0

    return noise_poselet

if __name__ == "__main__":
    all_data, _, file_names = load_samples(kActionRoot)

    print(all_data[0].shape)

    num_files = len(all_data)
    for i in progressbar.progressbar(range(num_files)):

        all_data[i] = make_some_noise(all_data[i])
