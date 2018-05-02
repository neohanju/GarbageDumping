import numpy as np
import progressbar
import os
import re
import glob
from time import localtime, strftime


# loading data samples and parsing their parameters from their file name
def load_samples(data_path, num_data=None):

    action_data, action_info, action_file_name = [], [], []
    file_paths = glob.glob(os.path.join(data_path, '*.npy'))

    print('Load data...')
    num_files = len(file_paths) if num_data is None else num_data
    for i in progressbar.progressbar(range(num_files)):
        # sample data
        action_data.append(np.load(file_paths[i]))

        # file names
        file_name = os.path.basename(file_paths[i]).split('.')[0]
        action_file_name.append(file_name)

        # sample information
        file_infos = re.split('[-.]+', file_name)
        file_infos = file_infos[0:-1]
        action_info.append(file_infos)

    return np.expand_dims(action_data, axis=3), action_info, action_file_name


def save_samples(save_path, result_data, file_names):
    os.mkdir(save_path)
    print('Save network outputs...')
    for i in progressbar.progressbar(range(len(file_names))):
        full_file_path = os.path.join(save_path, file_names[i] + "-recon.npy")
        np.save(full_file_path, np.squeeze(result_data[i], axis=2))


def get_time_string():
    strftime("%y%m%d-%H%M%S", localtime())


# ()()
# ('') HAANJU.YOO
