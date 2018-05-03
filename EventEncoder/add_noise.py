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
kActionLen = 30
kKeypointType = 14
nosing_rate = 0.1
nNoisePoint = int(kKeypointType * kActionLen * nosing_rate)


def generation_dirty_samples(_poselet ):

    noise_index = random.shuffle(list(range(0, kKeypointType * kActionLen)))

    start_idx = 0
    end_idx = nNoisePoint
    n_iter = int(kKeypointType * kActionLen / nNoisePoint)
    for i in range(n_iter):
        for idx in range(start_idx, end_idx):
            noise_poselet = copy.deepcopy(_poselet)
            frame_idx = int(idx/kKeypointType)
            point_idx = int(idx%kKeypointType)

            print(frame_idx, point_idx)

            noise_poselet[frame_idx][2 * point_idx + 0] = 0
            noise_poselet[frame_idx][2 * point_idx + 1] = 0


        #np.save()

        #start_idx += nNoisePoint
        #end_idx += nNoisePoint

    #return noise_poselet

if __name__ == "__main__":
    all_data, _, file_names = load_samples(kActionRoot)

    #num_files = len(all_data)
    num_files = 1
    for i in progressbar.progressbar(range(num_files)):

        generation_dirty_samples(all_data[i])

