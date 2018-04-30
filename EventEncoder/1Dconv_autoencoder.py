########################################
#     import requirement libraries     #
########################################
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Conv2D,  Conv2DTranspose
from keras.layers.normalization import BatchNormalization
from keras.backend import set_session

import numpy as np
import keras.layers
import keras.callbacks

import progressbar
import os
import re
import glob

# # set the quantity of GPU memory consumed
# import tensorflow as tf
# config = tf.ConfigProto()
#
# # use GPU memory in the available GPU memory capacity
# config.gpu_options.allow_growth = True
# sess = tf.Session(config=config)
# set_session(sess)

# weight initializer
from keras.initializers import glorot_uniform

kBasePath = "/home/neohanju/Workspace/dataset/etri_action_data/60_10"
# kResultVideoBasePath = os.path.join(kBasePath, "point_check")
kActionSampleBasePath = kBasePath
kInputSampleShape = (60, 36, 1)

#########################################################
#                   tensorboard setup                   #
#########################################################
tbCallBack = keras.callbacks.TensorBoard(log_dir='./Graph',
                                         histogram_freq=0, write_graph=True, write_images=True)

mcCallBack = keras.callbacks.ModelCheckpoint('./model/{epoch:05d}_epoch.hdf5', monitor='acc',
                                             save_best_only=True, period=50)


def parsing_file_name(_data_path):

    _action_data = []
    _action_info = []

    file_name_list = glob.glob(os.path.join(_data_path, '*.npy'))
    # num_total_file = len(file_name_list)
    num_total_file = 3
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
# input_poses = Input(shape=(60, 36, 1))
#
# x = Conv2D(1024, (5, 36), activation='relu')(input_poses)
# x_ = Conv2DTranspose(1, (5, 36), activation='relu')(x)

train_data = np.expand_dims(action_data, axis=3)

# model = Model(inputs=input_poses, outputs=x_)

num_filters_conv1 = 256
num_filters_conv2 = 128
num_z = 1024
size_kernle_conv1 = 36
size_kernel_conv2 = 10

size_kernle_conv3 = kInputSampleShape[0] - (size_kernle_conv1 - 1) - (size_kernel_conv2 - 1)

model = Sequential()
model.add(Conv2D(num_filters_conv1, (size_kernle_conv1, 36), input_shape=kInputSampleShape))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Conv2D(num_filters_conv2, (size_kernel_conv2, 1)))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Conv2D(num_z, (16, 1)))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Conv2DTranspose(num_filters_conv2, (size_kernle_conv3, 1)))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Conv2DTranspose(num_filters_conv1, (size_kernel_conv2, 1)))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Conv2DTranspose(1, (size_kernle_conv1, 36)))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.compile(optimizer='rmsprop', loss='mse', metrics=['accuracy'])
model.fit(train_data, train_data, epochs=10000, callbacks=[tbCallBack, mcCallBack])

