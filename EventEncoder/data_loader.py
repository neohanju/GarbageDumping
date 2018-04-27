import numpy as np
import os


class DataLoader():
    def __init__(self, x_data, y_data, batch_size=16, shuffle=True):
        self.data_root = data_root
        self.x_data = x_data
        self.y_data = y_data
        self.index = None
        self.batch_size = batch_size
        self.shuffle = shuffle
        # TODO: load all data code

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

if __name__ == "__main__":
