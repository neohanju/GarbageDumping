import numpy as np
import random
import json
import os
import copy
from xml.etree.ElementTree import ElementTree, parse, dump, Element, SubElement


params = {'step':      10,
          'interval':  60,
          'threshold': 30,
          'posi_label': 1,
          'bDrawGraph': True,
          'bUsingDisparity': False,
          'bNorm': True
          }


class MakeAction():
    def __init__(self, _save_dir_path, _track_root_path, _gt_path):
        self.save_dir_path = _save_dir_path
        self.track_root_path = _track_root_path
        self.ground_truth_path = _gt_path
        self.ground_truth = None

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

    def read_ground_truth(self, _cur_file):
        f = open(self.ground_truth_path, 'r')
        for line in f.readlines():
            split_line = line.split(' ')

            if int(split_line[0]) != _cur_file:
                continue

            ground_truth = []
            for dat in split_line[1:-1]:
                ground_truth.append(int(dat)) # 처음두개 값은 시작과 끝 frame 나머지는 box정보

        return ground_truth

    def pose_labeled(self, input_data, _gt):

        for id in input_data.keys():
            for frame_num in input_data[id].keys():
                cur_data = input_data[id][frame_num]
                neck = [cur_data[2], cur_data[3]]

                #GT에 걸리는 사람이면
                if _gt[0] <= frame_num <= _gt[1] and self.check_in_box(neck_coord=neck, _gt_box=_gt[2:]):
                    input_data[id][frame_num].append(1)
                    
                else:
                    input_data[id][frame_num].append(0)

        return input_data

    @staticmethod
    def check_in_box(neck_coord, _gt_box):
        if _gt_box[0] <= neck_coord[0] <= _gt_box[0] + _gt_box[1] and \
            _gt_box[2] <= neck_coord[1] <= _gt_box[2] + _gt_box[3]:
            return True

        else:
            return False

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
            print(frame_key)
            while 1:
                print(frame_key[start])

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
                    tmp_list = self.normalize_pose_(tmp_list, first_frame_neck, first_frame_dist)
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
                print("shape", action.shape)
                # print(action)
                self.save_action_npy(action, save_file_name)

                start += params['step']
                end += params['step']

        return action_data, sample_info

    @staticmethod
    def normalize_pose_(_pose_data, _neck, norm_constant):

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

    """
    def run(self):

        action = []
        info = []
        for file_num in self.file_num_list:

            # 이미 labeled data를 만들어 뒀으므로, 앞에 전처리 필요없이 labeled data를 불러오면 됨
            
            gt = self.read_ground_truth(file_num)
            track_data = self.read_track_data(file_num)
            labeled_data = self.pose_labeled(track_data, gt)
            tmp_action, tmp_info = self.make_action_data(file_num, labeled_data)

            if not action:
                action = tmp_action
                info = tmp_info
                continue
            action.extend(tmp_action)
            info.extend(tmp_info)

        return action, info
    """
    def run(self):
        action = []
        info = []
        for file_name in os.listdir(self.track_root_path):

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


def distance(v1,v2):
    return sum([(x-y)**2 for (x,y) in zip(v1,v2)])**(0.5)

class DataMaker():
    def __init__(self, _json_dir_path, _xml_dir_path, _save_dir_path, _file_num_list):
        self.json_dir_path = _json_dir_path
        self.xml_dir_path = _xml_dir_path
        self.save_dir_path = _save_dir_path
        self.file_num_list = _file_num_list

    def __del__(self):
        pass

    def read_json_pose_(self, _file_num, _frame_num):
        json_file_path = self.json_dir_path + "\\%03d\\%03d_%012d_keypoints.json" % (_file_num, _file_num, _frame_num)
        f = open(json_file_path, 'r')
        js = json.loads(f.read())
        f.close()

        return js

    @staticmethod
    def check_pose_in_gtbox_(_key_point, _attr, _margin=0):

        if (int(_attr['X']) - _margin <= _key_point[3] <= int(_attr['X']) + int(_attr['W']) + _margin and
                            int(_attr['Y']) - _margin <= _key_point[4] <= int(_attr['Y']) + int(_attr['H']) + _margin) and \
                ((int(_attr['X']) - _margin <= _key_point[6] <= int(_attr['X']) + int(_attr['W']) + _margin and
                                  int(_attr['Y']) - _margin <= _key_point[7] <= int(_attr['Y']) + int(_attr['W']) + _margin) or \
                                         int(_attr['X']) - _margin <= _key_point[15] <= int(_attr['X']) + int(_attr['W']) + _margin and
                                         int(_attr['Y']) - _margin <= _key_point[16] <= int(_attr['Y']) + int(_attr['W']) + _margin):
            return True

        else:
            return False

    # @staticmethod
    def packaging_preprocess_data_(self, _key_point, _label, _object, _attr):  # , _normalize, _scaling):
        result_data = []
        point = _key_point
        result_data.append(_object.find('ID').text)
        result_data.append(_object.find('Type').text)
        result_data.append(_attr['frameNum'])

        for i in range(18):
            result_data.append(str(point[i * 3]))
            result_data.append(str(point[i * 3 + 1]))
            result_data.append(str(point[i * 3 + 2]))
        result_data.append(str(_label))

        return result_data

    # labeling 된 데이터 저장하는 부분
    def saving_preprocess_data_(self, _list_data, _file_num):
        file_name = "%06d.txt" % _file_num
        save_file_path = self.save_dir_path + "\\%s" % file_name

        if file_name in os.listdir(self.save_dir_path):
            f = open(save_file_path, 'a')

        else:
            f = open(save_file_path, 'w')

        iter = 1
        for dat in _list_data:
            print(dat)
            f.write(dat)

            if len(_list_data) == iter:
                f.write("\n")
                continue

            f.write(" ")
            iter += 1

        f.close()

    def preprocess_data_(self, _ground_truth="xml"):
        for file_number in self.file_num_list:
            xml_file_path = self.xml_dir_path + "\\%03d.xml" % file_number

            # xml 파일 읽어오기
            tree = parse(xml_file_path)
            objects = tree.getroot().find('Objects')
            for object in objects:
                if not int(object.find('Type').text) in [1, 111]:
                    continue

                tracks = object.find('Tracks')
                for track in tracks.findall('Track'):
                    attr = track.attrib

                    # key point 읽어오기
                    people = self.read_json_pose_(file_number, int(attr['frameNum']))['people']

                    # 사람별 key point 접근
                    for person in people:
                        key_point = person['pose_keypoints']

                        # 주요한 점들이 박스 밖으로 나간 경우에는 해당 Pose 를 없애기
                        if not self.check_pose_in_gtbox_(key_point, attr):
                            continue

                        label = 0
                        if int(object.find('Type').text) == 111:
                            if _ground_truth == "macro":
                                if check_macro_file(file_number, int(attr['frameNum']),
                                                    int(object.find('ID').text)):  # positive frame 확인
                                    label = 1

                            else:
                                if check_verb_file(file_number, int(attr['frameNum']), key_point):
                                    label = 1

                        packaging_data = self.packaging_preprocess_data_(key_point, label, object, attr)
                        self.saving_preprocess_data_(packaging_data, file_number)


def check_verb_file(_file_number, _frame_number, _key_point):
    gt_root = "C:\\Users\\JM\\Desktop\\Data\\ETRIrelated\\BMVC\\ground_truth\\"
    gt_file_name = "%03d_gt.txt" % _file_number

    if gt_file_name not in os.listdir(gt_root):
        return False

    gt_path = gt_root + gt_file_name
    f = open(gt_path, 'r')
    result = False
    for lines in f.readlines():
        split_line = lines.split(' ')

        if _frame_number < int(split_line[0]) or _frame_number > int(split_line[1]):
            continue

        gt_box = [int(split_line[2]), int(split_line[3]), int(split_line[4]), int(split_line[5])]
        if check_ground_truth_box(_key_point, gt_box):
            result = True

    f.close()
    return result


def check_ground_truth_box(_key_point, _gt_box):
    num = 0
    for i in range(18):
        x = _key_point[3*i]
        y = _key_point[3*i +1]

        if _gt_box[0] > x or _gt_box[0] + _gt_box[2] < x:
            continue

        if _gt_box[1] > y or _gt_box[1] + _gt_box[3] < y:
            continue

        num += 1

    if num > 8:
        return True
    else:
        return False


class DataLoader():
    def __init__(self, batch_size=16, shuffle=True):
        self.x_data = None
        self.y_data = None
        self.index = None
        self.batch_size = batch_size
        self.shuffle = shuffle
        # TODO: load all data code
        pass

    def __getitem__(self, index):

        return self.x_data[index], self.y_data[index]

    def set_index(self):
        self.index = range(len(self.x_data))

        if self.shuffle:
            self.index = random.shuffle(self.index)

    def next_batch(self):

        if len(self.index) < self.batch_size:
            x_batch = self.x_data[np.asarray(self.index)]
            y_batch = self.y_data[np.asarray(self.index)]
            self.set_index()

        else:
            x_batch = self.x_data[np.asarray(self.index[0: self.batch_size])]
            y_batch = self.y_data[np.asarray(self.index[0: self.batch_size])]
            self.index = self.index[self.batch_size:]

        return x_batch, y_batch


def make_ground_truth():
    xml_path = "C:\\Users\\JM\\Desktop\\Data\\ETRIrelated\\final_xml"
    for file_name in os.listdir(xml_path):
        full_file_path = xml_path + "\\" + file_name
        file_number = int(file_name.split('.')[0])

        tree = parse(full_file_path)
        verbs = tree.getroot().find('Verbs')

        if not verbs:
            continue

        ground_truth = []
        for verb in verbs:

            if not int(verb.find('Type').text) == 200:
                continue

            boxes = []
            tracks = verb.find('Tracks')
            for track in tracks.findall('Track'):
                attr = track.attrib
                box = [int(attr['X']), int(attr['Y']), int(attr['W']), int(attr['H'])]
                boxes = box
                break

            if not (verb.find('StartFrame').text or verb.find('EndFrame').text):
                continue

            ground_truth.append([int(verb.find('StartFrame').text),
                                 int(verb.find('EndFrame').text)])

            ground_truth[-1].extend(boxes)

        #print(ground_truth)
        writing_text_file(file_number, ground_truth)


def writing_text_file(_file_number, _list):
    save_path = "C:\\Users\\JM\\Desktop\\Data\\ETRIrelated\\BMVC\\ground_truth\\" + "%03d_gt.txt" % _file_number
    f = open(save_path, 'w')

    l = 0
    for i, dat in enumerate(_list):
        for j, element in enumerate(dat):
            tmp = '%d' % element
            f.write(tmp)

            if len(dat) == j+1:
                f.write("\n")
                continue

            f.write(" ")
    f.close()





"""
# bending pose
files = [5, 15, 17, 18, 28,
         31, 41, 42, 58, 100,
         113, 115, 117, 124,
         125, 127, 133, 136, 140,
         144, 147,  160, 161,
         164, 165, 172, 186, 191,
         196, 199, 202, 207, 213
         ]
# 153 120 172,
frame = [703, 319, 278, 224, 755,
         1037, 871, 1442, 1761, 288,
         170, 269, 1049, 214,
         408, 499, 364, 202, 329,
         359, 254,  314, 135,
         369, 269, 839, 628, 522,
         715, 1194, 176, 352, 252]
# 419, 99 839,
"""

"""
# class 2
files = [6, 23, 13, 46, 57,
         61, 69, 95, 99]

frame = [593, 509, 329, 639, 824,
         134, 486, 319, 599]
"""

# class 3
files = [214, 195, 192, 188, 187,
         184, 183, 157, 123]

frame = [344, 271, 279, 367, 556,
         1596, 905, 409, 219]

if __name__ == "__main__":

    """
    f_list = [1, 6, 8, 9, 10, 11, 12, 13, 14, 15, 18, 22, 24, 26, 30, 32, 33, 34, 39, 41, 43, 46, 47, 49, 50, 51, 52, 53,
              56, 57, 59, 60, 61, 62, 63, 65, 67, 69, 70, 72, 79, 80, 81, 84, 85, 88, 94, 98, 101, 102, 104, 109, 110, 113,
              114, 115, 116, 118, 119, 120, 123, 124, 127, 129, 131, 133, 135, 136, 137, 139, 140, 141, 146, 147, 148, 149,
              151, 154, 155, 159, 160, 161, 162, 163, 164, 165, 168, 171, 172, 176, 181, 183, 184, 185, 186, 187, 188, 189,
              190, 191, 192, 193, 195, 196, 197, 199, 200, 201, 202, 203, 204, 205, 206, 208, 209, 210, 211, 212, 213, 214,
              217]
    """
    # make_ground_truth()
    """
    # read data
    xml_dir_path = "C:\\Users\\JM\\Desktop\\Data\\ETRIrelated\\final_xml"
    json_dir_path = "D:\\etri_data\\pose"
    save_dir_path = "C:\\Users\\JM\\Desktop\\Data\\ETRIrelated\\BMVC\\re_pose_data"

    loader = DataMaker(json_dir_path, xml_dir_path, save_dir_path, f_list)

    if not os.listdir(save_dir_path):
        loader.preprocess_data_()
    """

    save_action_path = "C:\\Users\\JM\\Desktop\\Data\\ETRIrelated\\BMVC\\action_data"
    pose_file_root = "C:\\Users\\JM\\Desktop\\Data\\ETRIrelated\\BMVC\\processed_keypoints"
    action_loader = MakeAction(save_action_path, pose_file_root, "")
    data, info = action_loader.run()

    # print(data[0])
