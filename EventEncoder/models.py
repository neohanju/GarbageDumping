from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Conv2D,  Conv2DTranspose, Input, Lambda, Dropout, Flatten
from keras.layers.normalization import BatchNormalization
import keras.backend as K
from keras import regularizers
from keras import metrics

kDefaultInputShape = (30, 28, 1)


def ConvAE(num_filters, kernel_sizes, num_z, input_shape=kDefaultInputShape, dropout=False):

    assert (len(num_filters) == len(kernel_sizes))

    last_kernel_size = input_shape[0]
    for sk in kernel_sizes:
        last_kernel_size -= (sk - 1)
        assert(0 < last_kernel_size)

    num_layers = len(num_filters)

    # encoder
    encoder_settings = [(num_filters[i], kernel_sizes[i]) for i in range(1, num_layers)]

    encoder = Sequential()
    encoder.add(Conv2D(num_filters[0], (kernel_sizes[0], input_shape[1]), input_shape=input_shape))
                      # kernel_regularizer=regularizers.l2(0.01),
                      # activity_regularizer=regularizers.l1(0.01)))
    encoder.add(BatchNormalization())
    encoder.add(Activation('relu'))
    for (num_fs, sz_k) in encoder_settings:
        encoder.add(Conv2D(num_fs, (sz_k, 1)))
        encoder.add(BatchNormalization())
        encoder.add(Activation('relu'))
        if dropout:
            encoder.add(Dropout(0.3))

    # output layer of encoder
    encoder.add(Conv2D(num_z, (last_kernel_size, 1)))
    encoder.add(BatchNormalization())
    encoder.add(Activation('linear', name='latent'))
    if dropout:
        encoder.add(Dropout(0.3))

    # encoder.summary()

    # for deconvolutional layers
    dec_num_filters = num_filters[::-1] + [input_shape[2]]
    dec_kernel_sizes = [last_kernel_size] + kernel_sizes[::-1]
    dec_channels = [1 if i < num_layers else input_shape[1] for i in range(num_layers + 1)]
    decoder_settings = [(dec_num_filters[i], dec_kernel_sizes[i], dec_channels[i]) for i in range(len(dec_num_filters))]

    decoder = Sequential()
    # decoder
    for i, (num_fs, sz_k, ch) in enumerate(decoder_settings):
        if 0 == i:
            decoder.add(Conv2DTranspose(num_fs, (sz_k, ch), input_shape=(1, 1, num_z)))
        else:
            decoder.add(Conv2DTranspose(num_fs, (sz_k, ch)))
        decoder.add(BatchNormalization())
        if i + 1 == len(decoder_settings):
            decoder.add(Activation('linear'))
        else:
            decoder.add(Activation('relu'))
            if dropout:
                decoder.add(Dropout(0.3))

    # decoder.summary()

    # for denoising encoder
    x = Input(shape=input_shape)
    z = encoder(x)
    x_hat = decoder(z)

    autoencoder = Model(inputs=x, outputs=x_hat)
    autoencoder.summary()

    return autoencoder, encoder, decoder


def ConvAE_Classifier(num_filters, kernel_sizes, num_z, input_shape=kDefaultInputShape, dropout=False):

    assert (len(num_filters) == len(kernel_sizes))

    last_kernel_size = input_shape[0]
    for sk in kernel_sizes:
        last_kernel_size -= (sk - 1)
        assert(0 < last_kernel_size)

    num_layers = len(num_filters)

    # encoder
    encoder_settings = [(num_filters[i], kernel_sizes[i]) for i in range(1, num_layers)]

    encoder = Sequential()
    encoder.add(Conv2D(num_filters[0], (kernel_sizes[0], input_shape[1]), input_shape=input_shape))
                      # kernel_regularizer=regularizers.l2(0.01),
                      # activity_regularizer=regularizers.l1(0.01)))
    encoder.add(BatchNormalization())
    encoder.add(Activation('relu'))
    for (num_fs, sz_k) in encoder_settings:
        encoder.add(Conv2D(num_fs, (sz_k, 1)))
        encoder.add(BatchNormalization())
        encoder.add(Activation('relu'))
        if dropout:
            encoder.add(Dropout(0.3))

    # output layer of encoder
    encoder.add(Conv2D(num_z, (last_kernel_size, 1)))
    encoder.add(BatchNormalization())
    encoder.add(Activation('linear', name='latent'))
    if dropout:
        encoder.add(Dropout(0.3))

    # encoder.summary()

    # for deconvolutional layers
    dec_num_filters = num_filters[::-1] + [input_shape[2]]
    dec_kernel_sizes = [last_kernel_size] + kernel_sizes[::-1]
    dec_channels = [1 if i < num_layers else input_shape[1] for i in range(num_layers + 1)]
    decoder_settings = [(dec_num_filters[i], dec_kernel_sizes[i], dec_channels[i]) for i in range(len(dec_num_filters))]

    decoder = Sequential()
    # decoder
    for i, (num_fs, sz_k, ch) in enumerate(decoder_settings):
        if 0 == i:
            decoder.add(Conv2DTranspose(num_fs, (sz_k, ch), input_shape=(1, 1, num_z)))
        else:
            decoder.add(Conv2DTranspose(num_fs, (sz_k, ch)))
        decoder.add(BatchNormalization())
        if i + 1 == len(decoder_settings):
            decoder.add(Activation('linear'))
        else:
            decoder.add(Activation('relu'))
            if dropout:
                decoder.add(Dropout(0.3))

    # decoder.summary()

    # classifier
    classifier = Sequential()
    classifier.add(Flatten(input_shape=(1, 1, num_z)))
    classifier.add(Dense(int(num_z * 0.5)))
    classifier.add(BatchNormalization())
    classifier.add(Activation('relu'))
    if dropout:
        classifier.add(Dropout(0.3))

    classifier.add(Dense(int(num_z * 0.25)))
    classifier.add(BatchNormalization())
    classifier.add(Activation('relu'))
    if dropout:
        classifier.add(Dropout(0.3))

    classifier.add(Dense(1))
    classifier.add(BatchNormalization())
    classifier.add(Activation('sigmoid'))


    # for denoising encoder
    x = Input(shape=input_shape)
    z = encoder(x)
    x_hat = decoder(z)
    label = classifier(z)

    autoencoder = Model(inputs=x, outputs=[x_hat, label])
    autoencoder.summary()

    return autoencoder, encoder, decoder


def vanila_autoencoder_loss(model_inputs, model_outputs):
    _, target_sample = model_inputs
    recon_sample, _ = model_outputs
    return float(K.shape(target_sample)[1] * K.shape(target_sample)[2]) * metrics.mse(target_sample, recon_sample)


def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=K.shape(z_mean), mean=0., stddev=1.0)
    return z_mean + K.exp(z_log_var / 2) * epsilon


def ConvVAE(num_filters, kernel_sizes, num_z, input_shape=kDefaultInputShape):

    assert (len(num_filters) == len(kernel_sizes))

    last_kernel_size = input_shape[0]
    for sk in kernel_sizes:
        last_kernel_size -= (sk - 1)
        assert(0 < last_kernel_size)

    num_layers = len(num_filters)

    # encoder
    encoder_settings = [(num_filters[i], kernel_sizes[i]) for i in range(1, num_layers)]

    encoder = Sequential()
    encoder.add(Conv2D(num_filters[0], (kernel_sizes[0], input_shape[1]), input_shape=input_shape))
    encoder.add(BatchNormalization())
    encoder.add(Activation('relu'))
    for (num_fs, sz_k) in encoder_settings:
        encoder.add(Conv2D(num_fs, (sz_k, 1)))
        encoder.add(BatchNormalization())
        encoder.add(Activation('relu'))

    # output layer of encoder
    encoder.add(Conv2D(num_z, (last_kernel_size, 1)))
    encoder.add(BatchNormalization())
    encoder.add(Activation('linear', name='latent'))

    encoder.summary()

    # latent space
    input_sample = Input(shape=input_shape)
    z_mean = encoder(input_sample)
    z_log_var = encoder(input_sample)
    z = Lambda(sampling)([z_mean, z_log_var])

    # for deconvolutional layers
    dec_num_filters = num_filters[::-1] + [input_shape[2]]
    dec_kernel_sizes = [last_kernel_size] + kernel_sizes[::-1]
    dec_channels = [1 if i < num_layers else input_shape[1] for i in range(num_layers + 1)]
    decoder_settings = [(dec_num_filters[i], dec_kernel_sizes[i], dec_channels[i]) for i in range(len(dec_num_filters))]

    # decoder
    decoder = Sequential()
    for i, (num_fs, sz_k, ch) in enumerate(decoder_settings):
        if i == 0:
            decoder.add(Conv2DTranspose(num_fs, (sz_k, ch), input_shape=(1, 1, num_z)))
        else:
            decoder.add(Conv2DTranspose(num_fs, (sz_k, ch)))
        if i + 1 == num_layers:
            decoder.add(Activation('linear'))
        else:
            decoder.add(Activation('relu'))

    # reconstruction
    recon_sample = decoder(z)
    target_sample = Input(shape=input_shape)

    decoder.summary()

    # instantiate VAE model
    vae = Model(inputs=[input_sample, target_sample], outputs=[recon_sample, z, z_mean, z_log_var])
    vae.summary()

    # Compute VAE loss
    mse_loss = float(input_shape[0]) * float(input_shape[1]) * metrics.mse(target_sample, recon_sample)
    kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
    vae_loss = K.mean(mse_loss + kl_loss)
    vae.add_loss(vae_loss)

    return vae


def vae_loss(model_inputs, model_outputs):
    _, target_sample = model_inputs
    recon_sample, z_mean, z_log_var = model_outputs
    mse_loss = float(K.shape(target_sample)[1] * K.shape(target_sample)[2]) * metrics.mse(target_sample, recon_sample)
    kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
    return mse_loss + kl_loss

# ()()
# ('')HAANJU.YOO
