import numpy as np
import random
import json
import os
import copy
import progressbar


params = {'step':      10,  # step은 한번 sample을 뽑고 몇 frame 이동 후에 뽑을지 결정합니다.
          'interval':  60,  # interval은 sample을 최대 몇 frame 연속으로 이어 붙일지를 결정합니다.
          'threshold': 30,  # sample을 만들 때 투기 pose가 threshold값 이상이라면 sample도 투기로 labeling합니다.
          'posi_label': 1
          }


kBasePath = "D:/workspace/data/BMVC/"
kKeypointBasePath = os.path.join(kBasePath, "processed_keypoints")
kSaveActionPath = os.path.join(kBasePath, "etri_action_data")


class MakeAction():
    def __init__(self, _save_dir_path, _track_root_path):
        self.save_dir_path = _save_dir_path
        self.track_root_path = _track_root_path
        # self.ground_truth_path = _gt_path
        # self.ground_truth = None

    def read_track_data(self, _cur_file):
        data = {}
        track_data_path = self.track_root_path + '/' + _cur_file

        f = open(track_data_path, 'r')
        for line in f.readlines():
            split_line = line.split(" ")

            if not int(split_line[0]) in data.keys():
                data[int(split_line[0])] = {}

            data[int(split_line[0])][int(split_line[1])] = []
            split_data = split_line[2:]

            for i, dat in enumerate(split_data):
                data[int(split_line[0])][int(split_line[2])].append(float(dat))

        return data

    def make_action_data(self, _file_number, pose_data):
        action_data = []
        sample_info = []
        for person_id in pose_data.keys():
            cur_person = pose_data[person_id]
            frame_key = list(cur_person.keys())
            frame_key.sort()

            # 액션을 만들만큼 충분한 pose가 있지 않은 경우
            if len(frame_key) < params['interval']:
                continue

            start = 0
            end = params['interval']
            # print(frame_key)
            while 1:
                # print(frame_key[start])

                # 액션의 끝 frame number가 존재하는 frame number 범위를 벗어나는 경우
                if end >= len(frame_key):
                    break

                if frame_key[end] != frame_key[start] + params['interval']:
                    start += 1
                    end += 1
                    continue
                    # break

                # sample 정보 저장(file number, pose 시작 frame number, pose 끝 frame number
                sample_info.append([_file_number, person_id, frame_key[start], frame_key[end]])

                # first frame info
                first_frame_neck = [cur_person[frame_key[start]][3 * 1 + 0], cur_person[frame_key[start]][3 * 1 + 1]]
                right_point = [cur_person[frame_key[start]][3 * 8 + 0], cur_person[frame_key[start]][3 * 8 + 1]]
                left_point = [cur_person[frame_key[start]][3 * 11 + 0], cur_person[frame_key[start]][3 * 11 + 1]]

                dist1 = distance(first_frame_neck, right_point)
                dist2 = distance(first_frame_neck, left_point)
                first_frame_dist = (dist1 + dist2) / 2

                label_check = 0
                # action_data.append([])
                action = []
                for i in frame_key[start:end]:

                    # print(len(cur_person[i]))
                    tmp_list = np.array(copy.deepcopy(cur_person[i]))

                    # 첫프레임 목좌표 0,0으로 만드는 좌표계로 변환!
                    # print("prev:", tmp_list)
                    tmp_list = self.normalize_pose(tmp_list, first_frame_neck, first_frame_dist)
                    # print("next:", tmp_list)

                    action.append([])
                    for j in range(18):
                        # action_data[-1].append(tmp_list[j])
                        action[-1].append(tmp_list[3 * j + 0])
                        action[-1].append(tmp_list[3 * j + 1])
                    
                    # action frame동안 투기로 labeled 된 pose가 몇갠지 세는 것
                    if cur_person[i][-1] == 1:
                        label_check += 1

                class_label = None
                # labeled 된것이 threshold 값보다 높다면 action을 투기action으로 labeling
                if label_check > params['threshold']:
                    # action_data[-1].append(1)
                    class_label = 1

                else:
                    # action_data[-1].append(0)
                    class_label = 0

                str_neck_x = format(first_frame_neck[0] + 100, '4.3f')
                str_neck_y = format(first_frame_neck[1] + 100, '4.3f')
                str_dist = format(first_frame_dist, '4.3f')

                str_neck_x = str_neck_x.replace('.', '_')
                str_neck_y = str_neck_y.replace('.', '_')
                str_dist = str_dist.replace('.', '_')
                save_file_name = "%03d-%02d-%04d-%03d-%02d-%s-%s-%s-%d.npy" \
                                 % (_file_number, person_id, frame_key[start], params['interval'], params['step'],
                                    str_neck_x, str_neck_y, str_dist, class_label)

                action = np.asarray(action)
                # print("shape", action.shape)
                # print(action)
                self.save_action_npy(action, save_file_name)

                start += params['step']
                end += params['step']

        return action_data, sample_info

    @staticmethod
    def normalize_pose(_pose_data, _neck, norm_constant):

        kXIdx = 0
        kYIdx = 1
        kConfidencIdx = 2
        kNumKeypointTypes = 18
        kOriginCoord = 100

        if isinstance(_neck, list):
            _neck = np.array(_neck)
        if isinstance(_pose_data, list):
            _pose_data = np.array(_pose_data)

        rescaled_origin = _neck[0:2] / norm_constant

        for base_index in range(kNumKeypointTypes):

            # 좌표가 (0,0) 인 것들을 가려내기 위해서 confidence 값을 사용한 것.
            pos_offset = 3 * base_index
            if _pose_data[pos_offset + kConfidencIdx] == 0:
                continue

            cur_point = _pose_data[pos_offset + kXIdx:pos_offset + kYIdx + 1]
            _pose_data[pos_offset + kXIdx:pos_offset + kYIdx + 1] = \
                cur_point / norm_constant - rescaled_origin + [kOriginCoord, kOriginCoord]

        return _pose_data

    def save_action_npy(self, _action, _save_file_name):
        save_file = self.save_dir_path + "\\" + _save_file_name
        np.save(save_file, _action)

    def read_labeled_data(self, _file_name):
        file_path = self.track_root_path + "\\" + _file_name

        data = {}
        f = open(file_path, 'r')
        for line in f.readlines():
            split_line = line.split(' ')

            if not int(split_line[0]) in data.keys():
                data[int(split_line[0])] = {}

            data[int(split_line[0])][int(split_line[2])] = []
            split_data = split_line[3:]

            for i, dat in enumerate(split_data):

                if len(split_data) == i + 1:
                    data[int(split_line[0])][int(split_line[2])].append(int(dat))
                    continue

                data[int(split_line[0])][int(split_line[2])].append(float(dat))

        return data

    def run(self):
        action = []
        info = []
        file_list = os.listdir(self.track_root_path)
        num_of_file = len(file_list)
        for i in progressbar.progressbar(range(num_of_file)):
            file_name = file_list[i]
            file_number = int(file_name.split(".")[0])
            labeled_data = self.read_labeled_data(file_name)
            tmp_action, tmp_info = self.make_action_data(file_number, labeled_data)

            if not action:
                action = tmp_action
                info = tmp_info
                continue
            action.extend(tmp_action)
            info.extend(tmp_info)

        return action, info


def distance(v1,v2):
    return sum([(x-y)**2 for (x,y) in zip(v1,v2)])**(0.5)


if __name__ == "__main__":

    action_loader = MakeAction(kSaveActionPath, kKeypointBasePath)
    data, info = action_loader.run()

    # print(data[0])
