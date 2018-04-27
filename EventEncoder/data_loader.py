import numpy as np


class DataLoader():
    def __init__(self, batch_size=16, shuffle=True):
        self.x_data = None
        self.y_data = None
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
