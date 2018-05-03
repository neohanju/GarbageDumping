import numpy as np
import progressbar
import os
import re
import glob
from utils import load_samples, save_samples

kDataRoot = "C:/Users/JM/Desktop/Data/ETRIrelated/BMVC"
kActionRoot = os.path.join(kDataRoot, "etri_action_data_neck_point_0_0")

if __name__ == "__main__":
    all_data, _, file_names = load_samples(kActionRoot)


