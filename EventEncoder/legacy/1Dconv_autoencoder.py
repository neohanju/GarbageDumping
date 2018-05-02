########################################
#     import requirement libraries     #
########################################
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Conv2D,  Conv2DTranspose
from keras.layers.normalization import BatchNormalization
from keras.backend import set_session
from keras import regularizers

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

kBasePath = "/home/jm/etri_action_data/30_10"
# kResultVideoBasePath = os.path.join(kBasePath, "point_check")
kActionSampleBasePath = kBasePath
kInputSampleShape = (30, 36, 1)

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
    num_total_file = len(file_name_list)
    # num_total_file = 3
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
size_kernel_conv1 = 18
size_kernel_conv2 = 5

size_kernel_conv3 = kInputSampleShape[0] - (size_kernel_conv1 - 1) - (size_kernel_conv2 - 1)

# num_filters = [2048, 1024, 512, 256, 128, 64]
# size_kernels = [  5,    5,   5,   5,   5,  5]
# #  output:       26,   22,  18,  14,  10,  6
# num_z = 128

num_filters = [256, 128]
size_kernels = [18,   5]
#  output:      13,   9
num_z = 1024


def n_layer_model(_num_fs, _sz_ks, _numz, _input_shape=kInputSampleShape):

    assert (len(_num_fs) == len(_sz_ks))

    _sz_z_k = _input_shape[0]
    for sk in _sz_ks:
        _sz_z_k -= (sk - 1)
        assert(0 < _sz_z_k)

    _num_layers = len(_num_fs)

    _model = Sequential()
    _model.add(Conv2D(_num_fs[0], (_sz_ks[0], 36),
                      input_shape=_input_shape))
    _model.add(BatchNormalization())
    _model.add(Activation('relu'))

    # for convolutional layers
    _conv_settings = [(_num_fs[i], _sz_ks[i]) for i in range(1, _num_layers)]
    _conv_settings.append((_numz, _sz_z_k))

    for (num_fs, sz_k) in _conv_settings:
        _model.add(Conv2D(num_fs, (sz_k, 1)))
        _model.add(BatchNormalization())
        _model.add(Activation('relu'))

    # for deconvolutional layers
    _num_fs = _num_fs[::-1]
    _num_fs.append(1)
    _sz_ks.append(_sz_z_k)
    _sz_ks = _sz_ks[::-1]
    _deconv_settings = [(_num_fs[i], _sz_ks[i]) for i in range(len(_num_fs))]
    _input_width = 1
    for i, (num_fs, sz_k) in enumerate(_deconv_settings):
        if i + 1 == len(_deconv_settings):
            _input_width = 36
        _model.add(Conv2DTranspose(num_fs, (sz_k, _input_width)))
        _model.add(BatchNormalization())
        if i+1 == len(_deconv_settings):
            # _model.add(Activation('tanh'))
            pass
        else:
            _model.add(Activation('relu'))

    return _model


def n_layer_reg_model(_num_fs, _sz_ks, _numz, _input_shape=kInputSampleShape):

    assert (len(_num_fs) == len(_sz_ks))

    _sz_z_k = _input_shape[0]
    for sk in _sz_ks:
        _sz_z_k -= (sk - 1)
        assert(0 < _sz_z_k)

    _num_layers = len(_num_fs)

    _model = Sequential()
    _model.add(Conv2D(_num_fs[0], (_sz_ks[0], 36),
                      kernel_regularizer=regularizers.l2(0.01),
                      activity_regularizer=regularizers.l1(0.01),
                      input_shape=_input_shape))
    _model.add(BatchNormalization())
    _model.add(Activation('relu'))

    # for convolutional layers
    _conv_settings = [(_num_fs[i], _sz_ks[i]) for i in range(1, _num_layers)]
    _conv_settings.append((_numz, _sz_z_k))

    for (num_fs, sz_k) in _conv_settings:
        _model.add(Conv2D(num_fs, (sz_k, 1),
                          kernel_regularizer=regularizers.l2(0.01),
                          activity_regularizer=regularizers.l1(0.01)))
        _model.add(BatchNormalization())
        _model.add(Activation('relu'))

    # for deconvolutional layers
    _num_fs = _num_fs[::-1]
    _num_fs.append(1)
    _sz_ks.append(_sz_z_k)
    _sz_ks = _sz_ks[::-1]
    _deconv_settings = [(_num_fs[i], _sz_ks[i]) for i in range(len(_num_fs))]
    _input_width = 1
    for i, (num_fs, sz_k) in enumerate(_deconv_settings):
        if i + 1 == len(_deconv_settings):
            _input_width = 36
        _model.add(Conv2DTranspose(num_fs, (sz_k, _input_width),
                                   kernel_regularizer=regularizers.l2(0.01),
                                   activity_regularizer=regularizers.l1(0.01)))
        _model.add(BatchNormalization())
        if i+1 == len(_deconv_settings):
            _model.add(Activation('tanh'))
        else:
            _model.add(Activation('relu'))

    return _model
# model = n_layer_model([num_filters_conv1, num_filters_conv2], [size_kernel_conv1, size_kernel_conv2], num_z)
# print(model)


def three_layer_model():

    _model = Sequential()
    _model.add(Conv2D(num_filters_conv1, (size_kernel_conv1, 36), input_shape=kInputSampleShape))
    _model.add(BatchNormalization())
    _model.add(Activation('relu'))

    _model.add(Conv2D(num_filters_conv2, (size_kernel_conv2, 1)))
    _model.add(BatchNormalization())
    _model.add(Activation('relu'))

    _model.add(Conv2D(num_z, (size_kernel_conv3, 1)))
    _model.add(BatchNormalization())
    _model.add(Activation('relu'))

    _model.add(Conv2DTranspose(num_filters_conv2, (size_kernel_conv3, 1)))
    _model.add(BatchNormalization())
    _model.add(Activation('relu'))

    _model.add(Conv2DTranspose(num_filters_conv1, (size_kernel_conv2, 1)))
    _model.add(BatchNormalization())
    _model.add(Activation('relu'))

    _model.add(Conv2DTranspose(1, (size_kernel_conv1, 36)))
    _model.add(BatchNormalization())
    _model.add(Activation('relu'))

    return _model

def fully_connected_model():

    _model = Sequential()
    _model.add(Dense())

    return _model

model = n_layer_model([num_filters_conv1, num_filters_conv2], [size_kernel_conv1, size_kernel_conv2], num_z)
model.compile(optimizer='rmsprop', loss='mse', metrics=['accuracy', 'mse'])
model.fit(train_data, train_data, epochs=10000, callbacks=[tbCallBack, mcCallBack], shuffle=True)

