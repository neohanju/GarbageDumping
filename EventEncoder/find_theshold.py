import argparse
import os
from utils import load_recon_error, calc_metric
import csv
import numpy as np
import progressbar

"""
# =============================================================================
# OPTIONS
# =============================================================================
parser = argparse.ArgumentParser(description='Autoencoding actions from keypoints')
# path related ---------------------------------------------------------------
parser.add_argument('--file_path', type=str,
                    default='/home/jm/workspace/GarbageDumping/EventEncoder/training_results/2018-05-08_00-24-41',
                    help='base path of reconstruction error file.')

options = parser.parse_args()
print(options)
"""
kResultBase = "/home/jm/workspace/GarbageDumping/EventEncoder/training_results/2018-05-08_00-24-41"
kReconErrorPath = os.path.join(kResultBase, "recon_error.csv")

def make_grid(list_of_mse):
    mse_min, mse_max = np.min(list_of_mse), np.max(list_of_mse)
    low = int(mse_min) + int(0.1*mse_max)
    high = int(mse_max) - int(0.1*mse_max)
    return range(low, high)



def find_threshold(threshold, _file_name_list, _label_list, _mse_list):
    all_info = []
    thresh_result = []

    for i, file_name in enumerate(_file_name_list):

        label = _label_list[i]
        mse = _mse_list[i]

        if _mse_list[i] >= threshold:
            thresh_result.append(1)
            all_info.append([file_name, label, mse, 1])

        else:
            thresh_result.append(0)
            all_info.append([file_name, label, mse, 0])

    res = calc_metric(_label_list, thresh_result)
    try:
        precision = res['tp'] / (res['tp'] + res['fp'])
        return precision

    except ZeroDivisionError:
        return np.nan


if __name__ == "__main__":

    recon_errors = load_recon_error(kReconErrorPath)
    mse_list = np.array(recon_errors[:,2],dtype=float)
    label_list = np.array(recon_errors[:,1], dtype=int)
    file_name_list = recon_errors[:,0]

    mse_grid = make_grid(mse_list)

    precision_list = []
    # print(recon_errors[0])
    for thres_candidate in progressbar.progressbar(mse_grid):
        precision = find_threshold(thres_candidate, file_name_list, label_list, mse_list)
        precision_list.append(precision)

    selected_grid_idx = np.nanargmax(np.asarray(precision_list))
    print("*"*50)
    print(mse_grid[selected_grid_idx])