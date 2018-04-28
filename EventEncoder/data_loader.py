import numpy as np
import os
import re
import random
import progressbar

kBasePath = "C:\\Users\\JM\\Desktop\\Data\\ETRIrelated\\BMVC\\"
# kResultVideoBasePath = os.path.join(kBasePath, "point_check")
kActionSampleBasePath = os.path.join(kBasePath, "action_data")


class DataLoader():
    def __init__(self, x_data, data_info, batch_size=16, shuffle=True):
        # self.data_root = data_root
        self.x_data = x_data
        # self.y_data = y_data
        self.data_info = data_info
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.index = None
        # self.blast
        # TODO: load all data code

    def __getitem__(self, index):

        return self.x_data[index], self.y_data[index]

    def set_index(self):
        data_index_list = list(range(len(self.x_data)))

        if self.shuffle:
            return random.shuffle(data_index_list)

        return data_index_list

    def next_batch(self):
        b_last = False

        if len(self.index) < self.batch_size:
            x_batch = self.x_data[np.asarray(self.index)]
            # y_batch = self.y_data[np.asarray(self.index)]
            info_batch = self.data_info[np.asarray(self.index)]
            self.set_index()
            b_last = True

        else:
            x_batch = self.x_data[np.asarray(self.index[0: self.batch_size])]
            # y_batch = self.y_data[np.asarray(self.index[0: self.batch_size])]
            info_batch = self.data_info[np.asarray(self.index[0: self.batch_size])]
            self.index = self.index[self.batch_size:]
            b_last = False

        return x_batch, info_batch, b_last


def read_data(_data_root):

    x = []
    y = []
    for file_name in os.listdir(_data_root):
        file_path = _data_root + "\\" + file_name
        tmp_name = file_name.split(".")[0]

        tmp_x = np.load(file_path)
        tmp_y = tmp_name.split("-")[-1]

        x.append(tmp_x)
        y.append(tmp_y)

    return x, y


def parsing_file_name(_data_path):

    _action_data = []
    _action_info = []

    file_name_list = os.listdir(_data_path)
    num_total_file = len(file_name_list)
    for i in progressbar.progressbar(range(num_total_file)):
        full_file_path = os.path.join(_data_path, file_name_list[i])
        _action_data.append(np.load(full_file_path))

        split_name = re.split('[-.]+', file_name_list[i])
        split_name = split_name[0:-1]
        _action_info.append(split_name)

    return _action_data, _action_info


if __name__ == "__main__":

    action_data, action_info = parsing_file_name(kActionSampleBasePath)

    loader = DataLoader(action_data, action_info, batch_size=16)
    idx = loader.set_index()
    print(idx)
    for epoch in range(1):
        while True:
            X_batch, Info_batch, is_last = loader.next_batch()

            if is_last:
                break

        print("exit")

    """
    all_person_index = []
    for sample in sample_list:
        split = re.split('[-.]+', sample)


    random.shuffle(all_person_index)
    train_person_index = all_person_index[0:160]
    print(train_person_index)
    print("\n")

    train_list = []
    for sample in sample_list:
        for train_person in train_person_index:
            if train_person == sample[0:6]:
                train_list.append(sample)

    print(len(sample_list))
    print(len(train_list))
    """