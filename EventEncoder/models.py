from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Conv2D,  Conv2DTranspose, Input
from keras.layers.normalization import BatchNormalization
from keras import regularizers

kDefaultInputShape = (30, 28, 1)

def ConvEncoder(num_filters, kernel_sizes, num_z, input_shape=kDefaultInputShape):
    assert (len(num_filters) == len(kernel_sizes))

    last_kernel_size = input_shape[0]
    for sk in kernel_sizes:
        last_kernel_size -= (sk - 1)
        assert (0 < last_kernel_size)
    num_layers = len(num_filters)

    # encoder
    encoder_settings = [(num_filters[i], kernel_sizes[i]) for i in range(1, num_layers)]
    encoder_settings.append((num_z, last_kernel_size))

    encoder = Sequential()
    encoder.add(Conv2D(num_filters[0], (kernel_sizes[0], input_shape[1]), input_shape=input_shape))
    encoder.add(BatchNormalization())
    encoder.add(Activation('relu'))
    for (num_fs, sz_k) in encoder_settings:
        encoder.add(Conv2D(num_fs, (sz_k, 1)))
        encoder.add(BatchNormalization())
        encoder.add(Activation('relu'))

    return encoder


def ConvDecoder(num_filters, kernel_sizes, num_z, output_shape=kDefaultInputShape):
    # TODO: check filter and kernel setting
    num_layers = len(num_filters)
    decoder_settings = [(num_filters[i], kernel_sizes[i]) for i in range(1, num_layers)]
    output_tensor_width = 1

    decoder = Sequential()
    decoder.add(Conv2DTranspose(num_filters[0], (kernel_sizes[0], 1), input_shape=(1, 1, num_z)))
    decoder.add(BatchNormalization())
    decoder.add(Activation('relu'))
    for i, (num_fs, sz_k) in enumerate(decoder_settings):
        if i + 1 == len(decoder_settings):
            output_tensor_width = output_shape[1]
        decoder.add(Conv2DTranspose(num_fs, (sz_k, output_tensor_width)))
        decoder.add(BatchNormalization())
        if i >= len(decoder_settings):
            decoder.add(Activation('relu'))

    return decoder


def ConvAE(num_filters, kernel_sizes, num_z, input_shape=kDefaultInputShape):

    assert (len(num_filters) == len(kernel_sizes))

    last_kernel_size = input_shape[0]
    for sk in kernel_sizes:
        last_kernel_size -= (sk - 1)
        assert(0 < last_kernel_size)

    num_layers = len(num_filters)

    # autoencoder
    encoder_settings = [(num_filters[i], kernel_sizes[i]) for i in range(1, num_layers)]

    autoencoder = Sequential()
    autoencoder.add(Conv2D(num_filters[0], (kernel_sizes[0], input_shape[1]), input_shape=input_shape))
                      # kernel_regularizer=regularizers.l2(0.01),
                      # activity_regularizer=regularizers.l1(0.01)))
    autoencoder.add(BatchNormalization())
    autoencoder.add(Activation('relu'))
    for (num_fs, sz_k) in encoder_settings:
        autoencoder.add(Conv2D(num_fs, (sz_k, 1)))
        autoencoder.add(BatchNormalization())
        autoencoder.add(Activation('relu'))
    # output layer of autoencoder
    autoencoder.add(Conv2D(num_z, (last_kernel_size, 1)))
    autoencoder.add(BatchNormalization())
    autoencoder.add(Activation('tanh', name='latent'))

    # for deconvolutional layers
    dec_num_filters = num_filters[::-1] + [input_shape[2]]
    dec_kernel_sizes = [last_kernel_size] + kernel_sizes[::-1]
    dec_channels = [1 if i < num_layers else input_shape[1] for i in range(num_layers + 1)]
    decoder_settings = [(dec_num_filters[i], dec_kernel_sizes[i], dec_channels[i]) for i in range(len(dec_num_filters))]

    # decoder
    for i, (num_fs, sz_k, ch) in enumerate(decoder_settings):
        autoencoder.add(Conv2DTranspose(num_fs, (sz_k, ch)))
        if i + 1 < num_layers:
            autoencoder.add(BatchNormalization())
            autoencoder.add(Activation('relu'))

    return autoencoder


def ConvVAE(num_filters, kernel_sizes, num_z, input_shape=kDefaultInputShape):

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

# ()()
# ('')HAANJU.YOO
