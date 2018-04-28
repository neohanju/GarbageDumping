########################################
#     import requirement libraries     #
########################################
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import Flatten
from keras.layers.normalization import BatchNormalization
from keras.layers import Merge, Input
from keras.layers.core import Dropout
from keras.backend import set_session
from keras.models import Model
from keras.activations import softmax
from keras.layers import Conv2D, Conv1D, MaxPooling1D, Conv2DTranspose

from keras.models import model_from_json
from keras.models import load_model
from keras.models import save_model
from keras.utils import np_utils
import keras.backend as K
import scipy.io as sio
import numpy as np
import keras.layers
from keras.utils.np_utils import to_categorical

from keras.utils.vis_utils import model_to_dot
import random
import matplotlib.pyplot as plt

# time stamp
import timeit

import data_loader
import progressbar
import os
import re
import random
import glob

# set the quantity of GPU memory consumed
import tensorflow as tf
config = tf.ConfigProto()
# use GPU memory in the available GPU memory capacity
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
set_session(sess)

# weight initializer
from keras.initializers import glorot_uniform

#########################################################
#                   tensorboard setup                   #
#########################################################
from keras.callbacks import TensorBoard


tbCallBack = keras.callbacks.TensorBoard(log_dir='./Graph',
                                         histogram_freq=0, write_graph=True, write_images=True)

kBasePath = "C:\\Users\\JM\\Desktop\\Data\\ETRIrelated\\BMVC\\"
# kResultVideoBasePath = os.path.join(kBasePath, "point_check")
kActionSampleBasePath = os.path.join(kBasePath, "action_data")


def parsing_file_name(_data_path):

    _action_data = []
    _action_info = []

    file_name_list = glob.glob(os.path.join(_data_path, '*.npy'))
    num_total_file = len(file_name_list)
    for i in progressbar.progressbar(range(num_total_file)):
        full_file_path = os.path.join(_data_path, file_name_list[i])
        _action_data.append(np.load(full_file_path))

        split_name = re.split('[-.]+', file_name_list[i])
        split_name = split_name[0:-1]
        _action_info.append(split_name)

    return _action_data, _action_info


action_data, action_info = parsing_file_name(kActionSampleBasePath)

print(action_data[0].shape)

# Encoder
input_poses = Input(shape=(60, 36, 1))

x = Conv2D(1024, (5, 36), activation='relu')(input_poses)
x_ = Conv2DTranspose(1, (5, 36), activation='relu')(x)

test_data = np.expand_dims(action_data, axis=3)

model = Model(inputs=input_poses, outputs=x_)
model.compile(optimizer='rmsprop', loss='mse', metrics=['accuracy'])
model.fit(test_data, test_data, epochs=100, callbacks=[tbCallBack])

