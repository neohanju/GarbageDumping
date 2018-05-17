import os
import glob
import re
import csv
import math
import numpy as np
import progressbar
from keras.models import load_model
from utils import load_recon_error, calc_metric, save_recon_error, load_samples

kDataBase = "/home/jm/workspace/dataset/etri_action_data"
kActionData = os.path.join(kDataBase, "30_10/etri")
kGTBase = os.path.join(kDataBase, "gt_csv")
kCVBase = os.path.join(kDataBase, "k_fold")
kTrainBase = "/home/jm/workspace/GarbageDumping/EventEncoder/training_results"
kDetectBase = os.path.join(kTrainBase, "2018-05-08_00-24-41") #TODO: change

def read_ground_truth(_video_name):
    csv_name = "%03d.csv" %int(_video_name)
    full_file_path = os.path.join(kGTBase, csv_name)

    result_list = []
    with open(full_file_path) as f:
        reader = csv.reader(f)
        for row in reader:
            result_list.append(row)

    return result_list[0], result_list[1:]

def cv_file_list(_k, cv_type='train'):
    assert(_k>=0)

    csv_name = "k-score%d.csv" % (_k)
    if not csv_name in glob.glob1(kCVBase, "*.csv"):
        print(csv_name, "is not exist")
        return False

    file_content = []
    full_file_path = os.path.join(kCVBase, csv_name)
    with open(full_file_path) as f:
        reader = csv.reader(f)
        for row in reader:
            file_content.append(row)

    if cv_type == 'train':
        return file_content[0]

    elif cv_type == 'test':
        return file_content[1]

    elif cv_type == 'validation':
        return file_content[2]

def check_detect_result(video_frame_info, _video_name, _thresh, experiment_type='test', interval=90):
    # csv_name = "%03d.csv" % int(_video_name)
    first_frame = int(video_frame_info[0])
    last_frame = int(video_frame_info[1])
    result_length = math.ceil(last_frame/interval)

    if experiment_type == 'train':
        full_file_path = os.path.join(kDetectBase, "recon_error.csv") # TODO: change detect csv
    elif experiment_type == 'test':
        full_file_path = os.path.join(kDetectBase, "recon_test_error.csv")
    elif experiment_type == 'validation':
        full_file_path = os.path.join(kDetectBase, "recon_val_error.csv")

    result = {}
    with open(full_file_path) as f:
        reader = csv.reader(f)
        for row in reader:

            split_name = re.split('[-.]+', row[0])
            video_num = int(split_name[0])
            if video_num != int(_video_name):
                continue

            track_id = int(split_name[1])
            start_frame = int(split_name[2])
            label = 1 if _thresh < float(row[-1]) else 0
            mse = float(row[-1])
            gt_label = int(row[-2])

            if track_id not in result.keys():
                result[track_id] = {}
                result[track_id]['gt'] = [0] * result_length
                result[track_id]['pos'] = [0] * result_length
                result[track_id]['mse'] = [0] * result_length
                result[track_id]['sample'] = [0] * result_length

            if gt_label == 1:
                result[track_id]['gt'][int(start_frame / interval)] += 1

            if label == 1:
                result[track_id]['pos'][int(start_frame/interval)] += 1
            result[track_id]['mse'][int(start_frame/interval)] += mse

            result[track_id]['sample'][int(start_frame/interval)] += 1

    pos_table = []
    mse_table = []
    gt_table = []
    for t_id in result.keys():
        avg_gt = []
        avg_mse = []
        avg_pos = []
        for i, num_sample in enumerate(result[t_id]['sample']):

            if num_sample == 0:
                avg_pos.append(0)  # / np.array(result[t_id]['sample'])
                avg_mse.append(0)  # / np.array(result[t_id]['sample'])
                avg_gt.append(0)
                continue

            avg_pos.append(0 if result[t_id]['pos'][i]/num_sample < 0.5 else 1) # final voting
            avg_mse.append(result[t_id]['mse'][i]/num_sample)
            avg_gt.append(0 if result[t_id]['gt'][i]/num_sample < 0.5 else 1)

        gt_table.append(avg_gt)
        pos_table.append(avg_pos)
        mse_table.append(avg_mse)

    return gt_table, pos_table, mse_table

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
        recall = res['tp'] / (res['tp'] + res['fn'])
        f1 = 2 * precision * recall / (precision + recall)
        return f1

    except ZeroDivisionError:
        return np.nan

def find_outer_thresh(_gt, _mse_pred):

    mse_min = 0
    mse_max = 100

    mse_grid = range(mse_min, mse_max)
    f1s_list = []
    for grid in mse_grid:
        tp = 0
        fn = 0
        tn = 0
        fp = 0

        tmp_pred = []
        tmp_gt = []
        for i in range(len(_mse_pred)):
            for j, tmp in enumerate(_mse_pred[i]):
                if tmp > grid:
                    tmp_pred.append(1)
                else:
                    tmp_pred.append(0)

                tmp_gt.append(_gt[i][j])


        res = calc_metric(tmp_gt, tmp_pred)
        tp += res['tp']
        fn += res['fn']
        tn += res['tn']
        fp += res['fp']

        try:
            precision = res['tp'] / (res['tp'] + res['fp'])
            recall = res['tp'] / (res['tp'] + res['fn'])
            f1 = 2 * precision * recall / (precision + recall)
            f1s_list.append(f1)

        except ZeroDivisionError:
            f1s_list.append(np.nan)

    return list(mse_grid)[int(np.nanargmax(f1s_list))]

if __name__ == "__main__":

    for k in range(10):

        # validation step for find threshold value
        video_list = cv_file_list(k, 'validation') # k = 1
        data_info, action_data, _ = load_samples(kActionData)
        val_data = []
        val_info = []
        for i, info in enumerate(data_info):
            video_name = str(int(info['video_name']))
            if video_name in video_list:
                val_data.append(action_data[i])
                val_info.append(info)
        val_data = np.array(val_data)

        # TODO: run model for make validation error
        model_name = os.path.join(kDetectBase, "best_loss.hdf5")
        model = load_model(model_name)
        predictions = model.predict(val_data)
        save_recon_error(kDetectBase, val_data, predictions, val_info, "recon_val_error.csv")

        # load validation reconstruction error
        val_recon_error_path = os.path.join(kDetectBase, "recon_val_error.csv")
        val_recon_error = load_recon_error(val_recon_error_path)

        mse_list = np.array(val_recon_error[:, 2], dtype=float)
        label_list = np.array(val_recon_error[:, 1], dtype=int)
        file_name_list = val_recon_error[:, 0]

        mse_grid = make_grid(mse_list)

        # find threshold using grid search method
        f1_list = []
        for thres_candidate in progressbar.progressbar(mse_grid):
            f1 = find_threshold(thres_candidate, file_name_list, label_list, mse_list)
            f1_list.append(f1)

        print("val procedure")
        print(f1_list)
        selected_grid_idx = np.nanargmax(np.asarray(f1_list))
        selected_threshold = mse_grid[selected_grid_idx]


        hole_gt = []
        hole_mse_pred = []
        for video_name in video_list:
            video_frame, gt_frame = read_ground_truth(video_name)
            gt, pred_pos, pred_mse = check_detect_result(video_frame, video_name, selected_threshold, experiment_type='validation') #TODO: find threshold using validation set
            hole_gt.extend(gt)
            hole_mse_pred.extend(pred_mse)

        outer_thresh = find_outer_thresh(hole_gt, hole_mse_pred)


        # final evaluation procedure
        video_list = cv_file_list(k, 'test')
        test_data = []
        test_info = []
        for i, info in enumerate(data_info):
            video_name = str(int(info['video_name']))
            if video_name in video_list:
                test_data.append(action_data[i])
                test_info.append(info)
        test_data = np.array(test_data)

        # run model for testing
        predictions = model.predict(test_data)
        save_recon_error(kDetectBase, test_data, predictions, test_info, "recon_test_error.csv")

        # load validation reconstruction error
        test_recon_error_path = os.path.join(kDetectBase, "recon_test_error.csv")
        test_recon_error = load_recon_error(test_recon_error_path)

        tp_pos = 0
        fn_pos = 0
        tn_pos = 0
        fp_pos = 0
        tp_mse = 0
        fn_mse = 0
        tn_mse = 0
        fp_mse = 0
        for video_name in video_list:
            video_frame, gt_frame = read_ground_truth(video_name)
            gt, pred_pos, pred_mse = check_detect_result(video_frame, video_name, selected_threshold, experiment_type='test') #TODO: find threshold using validation set
            #
            # print(video_name)
            # print(gt)
            # print(pred_mse)
            # print(pred_pos)
            # print('*'*50)
            #print(len(gt))

            for i in range(len(gt)):
                res = calc_metric(gt[i], pred_pos[i])
                tp_pos += res['tp']
                fn_pos += res['fn']
                tn_pos += res['tn']
                fp_pos += res['fp']

            for i in range(len(gt)):
                tmp_mse = []
                for j in range(len(pred_mse[i])):
                    tmp_mse.append(pred_mse[i][j])
                res = calc_metric(gt[i], tmp_mse)
                tp_mse += res['tp']
                fn_mse += res['fn']
                tn_mse += res['tn']
                fp_mse += res['fp']

        print("Test procedure")
        print('tp:', tp_pos, 'fp', fp_pos, 'fn', fn_pos, 'tn', tn_pos)
        print('precision:', tp_pos/(tp_pos + fp_pos))

        print("Test procedure")
        print('tp:', tp_mse, 'fp', fp_mse, 'fn', fn_mse, 'tn', tn_mse)
        print('precision:', tp_mse / (tp_mse + fp_mse))

#########################################

