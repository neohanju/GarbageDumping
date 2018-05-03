from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Conv2D,  Conv2DTranspose
from keras.layers.normalization import BatchNormalization
from keras import regularizers


def ConvVAE(num_filters, kernel_sizes, num_z, input_shape=(30, 36, 1)):

    assert (len(num_filters) == len(kernel_sizes))

    last_kernel_size = input_shape[0]
    for sk in kernel_sizes:
        last_kernel_size -= (sk - 1)
        assert(0 < last_kernel_size)

    num_layers = len(num_filters)

    encoder = Sequential()
    encoder.add(Conv2D(num_filters[0], (kernel_sizes[0], input_shape[1]), input_shape=input_shape))
    encoder.add(BatchNormalization())
    encoder.add(Activation('relu'))

    # for convolutional layers
    conv_layer_params = [(num_filters[i], kernel_sizes[i]) for i in range(1, num_layers)]
    conv_layer_params.append((num_z, last_kernel_size))

    for (num_filter, kernel_size) in conv_layer_params:
        encoder.add(Conv2D(num_filter, (kernel_size, 1)))
        encoder.add(BatchNormalization())
        encoder.add(Activation('relu'))

    #TODO: complete below

    # for deconvolutional layers
    num_filters = num_filters[::-1]
    num_filters.append(input_shape[2])
    kernel_sizes.append(last_kernel_size)
    kernel_sizes = kernel_sizes[::-1]
    _deconv_settings = [(num_filters[i], kernel_sizes[i]) for i in range(len(num_filters))]
    _input_width = 1
    for i, (num_filter, kernel_size) in enumerate(_deconv_settings):
        if i + 1 == len(_deconv_settings):
            _input_width = input_shape[1]
        encoder.add(Conv2DTranspose(num_filter, (kernel_size, _input_width)))
        encoder.add(BatchNormalization())
        if i + 1 == len(_deconv_settings):
            # encoder.add(Activation('tanh'))
            pass
        else:
            encoder.add(Activation('relu'))

    return encoder


def ConvAE(_num_fs, _sz_ks, _numz, _input_shape=(30, 36, 1)):

    assert (len(_num_fs) == len(_sz_ks))

    _sz_z_k = _input_shape[0]
    for sk in _sz_ks:
        _sz_z_k -= (sk - 1)
        assert(0 < _sz_z_k)

    _num_layers = len(_num_fs)

    _model = Sequential()
    _model.add(Conv2D(_num_fs[0], (_sz_ks[0], _input_shape[1]),
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
    _num_fs.append(_input_shape[2])
    _sz_ks.append(_sz_z_k)
    _sz_ks = _sz_ks[::-1]
    _deconv_settings = [(_num_fs[i], _sz_ks[i]) for i in range(len(_num_fs))]
    _input_width = 1
    for i, (num_fs, sz_k) in enumerate(_deconv_settings):
        if i + 1 == len(_deconv_settings):
            _input_width = _input_shape[1]
        _model.add(Conv2DTranspose(num_fs, (sz_k, _input_width)))
        _model.add(BatchNormalization())
        if i+1 == len(_deconv_settings):
            # _model.add(Activation('tanh'))
            pass
        else:
            _model.add(Activation('relu'))

    return _model


def ConvAE_regularization(_num_fs, _sz_ks, _numz, _input_shape=(30, 36, 1)):

    assert (len(_num_fs) == len(_sz_ks))

    _sz_z_k = _input_shape[0]
    for sk in _sz_ks:
        _sz_z_k -= (sk - 1)
        assert(0 < _sz_z_k)

    _num_layers = len(_num_fs)

    _model = Sequential()
    _model.add(Conv2D(_num_fs[0], (_sz_ks[0], _input_shape[1]),
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
    _num_fs.append(_input_shape[2])
    _sz_ks.append(_sz_z_k)
    _sz_ks = _sz_ks[::-1]
    _deconv_settings = [(_num_fs[i], _sz_ks[i]) for i in range(len(_num_fs))]
    _input_width = 1
    for i, (num_fs, sz_k) in enumerate(_deconv_settings):
        if i + 1 == len(_deconv_settings):
            _input_width = _input_shape[1]
        _model.add(Conv2DTranspose(num_fs, (sz_k, _input_width),
                                   kernel_regularizer=regularizers.l2(0.01),
                                   activity_regularizer=regularizers.l1(0.01)))
        _model.add(BatchNormalization())
        if i+1 == len(_deconv_settings):
            _model.add(Activation('tanh'))
        else:
            _model.add(Activation('relu'))

    return _model


def fully_connected_model():
    _model = Sequential()
    _model.add(Dense())

    return _model