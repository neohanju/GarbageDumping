import numpy as np
import progressbar
import os
import re
import glob
import copy
import csv
from time import localtime, strftime


def make_dir(path):
    # if there is no directory, make a directory.
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except OSError:
            print('ERROR: Cannot make saving folder')
    return


def parsing_action_file_name(file_name):
    # 000003_bonn-00-2121-030-10-738_516-336_083-131_216-0.npy
    # [VIDEO NAME]-[ID]-[FRAME #]-[SAMPLE LENGTH]-[SAMPLE INTERVAL]-[NECK X]_[NECK X]-[NECK Y]_[NECK Y]-[TORSO]_[TORSO]-[THROWING LABEL].npy

    file_name_part = os.path.basename(file_name).split('.')[0]  # ensure that 'file_name' does not have directory part
    file_info_strings = re.split('[-.]+', file_name_part)
    original_file_name = '-'.join(file_info_strings[0:9])  # only differ from 'file_name_part' when the sample is a noisy sample
    file_info_dict = {
        'file_name': file_name_part,
        'original_name': original_file_name,
        'video_name': file_info_strings[0],
        'track_id': int(file_info_strings[1]),
        'frame_number': int(file_info_strings[2]),
        'sample_length': int(file_info_strings[3]),
        'sample_interval': int(file_info_strings[4]),
        'neck_point': [float(file_info_strings[5].replace('_', '.')), float(file_info_strings[6].replace('_', '.'))],
        'chest_length': float(file_info_strings[7].replace('_', '.')),
        'label': int(file_info_strings[8]),
        'dirty_label': None if len(file_info_strings) <= 9 else file_info_strings[9]
    }
    return file_info_dict


def load_sample(sample_path):
    sample = np.load(sample_path)
    if sample.ndim < 3:
        sample = np.expand_dims(sample, axis=3)
    return sample


# loading data samples and parsing their parameters from their file name
def load_samples(data_path, dirty_sample_path=None, num_data=None):
    # return Xs, infos, targets (when dirty samples are ready)

    action_info, action_data = [], []
    file_paths = glob.glob(os.path.join(data_path, '*.npy'))
    file_paths.sort()

    print('Load data...')
    num_files = len(file_paths) if num_data is None else num_data
    for i in progressbar.progressbar(range(num_files)):
        # sample information
        action_info.append(parsing_action_file_name(file_paths[i]))

        # sample data
        action_data.append(load_sample(file_paths[i]))

    if dirty_sample_path is not None and num_data is None:
        assert(os.path.exists(os.path.join(data_path, dirty_sample_path)))
        dirty_file_paths = glob.glob(os.path.join(data_path, dirty_sample_path, '*.npy'))
        dirty_file_paths.sort()
        print('Load noisy data...')

        target_info = copy.deepcopy(action_info)
        target_data = [single_data for single_data in action_data]
        target_idx = 0
        for i in progressbar.progressbar(range(len(dirty_file_paths))):
            # sample information
            cur_info = parsing_action_file_name(dirty_file_paths[i])

            search_length = 0
            while target_info[target_idx]['file_name'] not in cur_info['file_name']:
                target_idx += 1
                search_length += 1
                if len(target_info) <= target_idx:
                    target_idx = 0
                if len(target_info) <= search_length:
                    print("There is no original data for " + cur_info['file_name'])
                    assert False

            # sample info
            action_info.append(cur_info)

            # sample data
            action_data.append(load_sample(dirty_file_paths[i]))

            # target data
            target_data.append(action_data[target_idx])
    else:
        # for reconstruction
        target_data = action_data

    return action_info, np.stack(action_data, axis=0), np.stack(target_data, axis=0)

    # return action_info, np.expand_dims(action_data, axis=3), np.expand_dims(target_data, axis=3)


def load_latent_vectors(data_path, num_data=None):

    latent_info, latent_data = [], []
    file_paths = glob.glob(os.path.join(data_path, '*.npy'))
    file_paths.sort()

    print('Load latent vectors...')
    num_files = len(file_paths) if num_data is None else num_data
    for i in progressbar.progressbar(range(num_files)):
        # sample information
        latent_info.append(parsing_action_file_name(file_paths[i]))

        # sample data
        latent_data.append(np.reshape(load_sample(file_paths[i]), (1, 1, -1)))

    return latent_info, np.stack(latent_data)


def load_recon_error(data_path):

    recon_error_list = []

    print('Load reconstruction errors...')
    with open(data_path, 'r') as f:
        lines = f.readlines()

        for i in progressbar.progressbar(range(len(lines))):
            split_line = lines[i].split(',')
            file_name = split_line[0]
            label = split_line[1]
            recon_error = split_line[2]
            recon_error_list.append([file_name, label, recon_error])

    return np.array(recon_error_list)

def combine_input_and_target(input_file_names, target_file_names):
    input_file_names.sort()
    target_file_names.sort()

    input_target_index_pairs = []

    input_idx = 0
    for i, file_name in enumerate(target_file_names):
        while input_idx < len(input_file_names) and \
                        file_name in input_file_names[input_idx]:
            input_target_index_pairs.append((input_idx, i))
            input_idx += 1

    return input_target_index_pairs


def save_samples(save_path, result_data, file_infos, postfix='-recon'):
    make_dir(save_path)
    print('Save network outputs...')
    for i in progressbar.progressbar(range(len(file_infos))):
        full_file_path = os.path.join(save_path, file_infos[i]['file_name'] + postfix + ".npy")
        np.save(full_file_path, np.squeeze(result_data[i], axis=2))


def save_latent_variables(save_path, latent_data, file_infos):
    make_dir(save_path)
    print('Save network outputs...')
    for i in progressbar.progressbar(range(len(file_infos))):
        full_file_path = os.path.join(save_path, file_infos[i]['file_name'] + "-latent.npy")
        np.save(full_file_path, latent_data[i].flatten())

def save_recon_error(save_path, input, prediction, file_infos, file_name='recon_error.csv'):
    full_file_path = os.path.join(save_path, file_name)
    print('Save reconstruction errors...')

    recon_error_list = []
    for i in progressbar.progressbar((range(len(file_infos)))):
        recon_errors = np.sum((input[i] - prediction[i]) ** 2) ** 0.5
        # print(file_infos[i])
        file_name = file_infos[i]['file_name']
        label = re.split('[-.]+', file_name)[-1]
        recon_error_list.append([file_name, label, float(recon_errors)])

    with open(full_file_path, "w") as f:
        writer = csv.writer(f)
        writer.writerows(recon_error_list)

def get_time_string():
    strftime("%y%m%d-%H%M%S", localtime())


def intersection_over_union(box1, box2):
    # box: [x1, y1, x2, y2]
    inter_x1, inter_y1 = max(box1[0], box2[0]), max(box1[1], box2[1])
    inter_x2, inter_y2 = min(box1[2], box2[2]), min(box1[3], box2[3])
    if inter_x1 > inter_x2 or inter_y1 > inter_y2:
        return 0.0
    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return inter_area / (area1 + area2 - inter_area)

def calc_metric(gt_list, detect_list):

    tmp_list = np.array(gt_list) + 2* np.array(detect_list)

    result = {'tp' : 0,
              'fn' : 0,
              'fp' : 0,
              'tn' : 0}
    tp = tn = fp = fn = 0
    for res in tmp_list:

        if res == 0:
            result['tn'] += 1

        elif res == 1:
            result['fn'] += 1

        elif res == 2:
            result['fp'] += 1

        else:
            result['tp'] += 1

    #precision = tp / tp + fp
    #recall = tp / tp + fn

    return result

# ()()
# ('') HAANJU.YOO
