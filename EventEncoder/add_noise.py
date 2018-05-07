import numpy as np
import copy
import progressbar
import os
import re
import glob
import random
from utils import load_samples, save_samples

kDataRoot = "C:/Users/JM/Desktop/Data/ETRIrelated/BMVC"
kActionRoot = os.path.join(kDataRoot, "etri_action_data")
kSavePath = "C:/Users/JM/Desktop/Data/ETRIrelated/BMVC/etri_dirty_sample"

kActionLen = 30
kKeypointType = 14
nosing_rate = 0.1
nNoisePoint = int(kKeypointType * kActionLen * nosing_rate)


def generation_dirty_samples(_poselet, _file_name):

    noise_index = list((range(0, kKeypointType * kActionLen)))
    random.shuffle(noise_index)

    start_idx = 0
    end_idx = nNoisePoint
    n_iter = int(kKeypointType * kActionLen / nNoisePoint)

    for i in range(n_iter):
        noise_poselet = copy.deepcopy(_poselet)

        for idx in range(start_idx, end_idx):
            frame_idx = int(noise_index[idx]/kKeypointType)
            point_idx = int(noise_index[idx]%kKeypointType)
            # print(frame_idx, point_idx)

            noise_poselet[frame_idx][2 * point_idx + 0] = 0
            noise_poselet[frame_idx][2 * point_idx + 1] = 0

        save_name = _file_name + "-d%d.npy" %i
        save_path = os.path.join(kSavePath, save_name)
        start_idx += nNoisePoint
        end_idx += nNoisePoint

        np.save(save_path, noise_poselet)
    #return noise_poselet

if __name__ == "__main__":
    all_data, _, file_names = load_samples(kActionRoot)

    num_files = len(all_data)
    # num_files = 1
    for i in progressbar.progressbar(range(num_files)):

        generation_dirty_samples(all_data[i], file_names[i])
