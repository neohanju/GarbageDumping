import argparse
import random
import os

from keras import callbacks

from models import ConvAE #, ConvVAE, vanila_autoencoder_loss, vae_loss
from utils import load_samples, get_time_string


# =============================================================================
# OPTIONS
# =============================================================================
parser = argparse.ArgumentParser(description='Autoencoding actions from keypoints')
# model related ---------------------------------------------------------------
parser.add_argument('--model', type=str, default='AE', help="types of model. one of 'AE | VAE'.")
parser.add_argument('--nfs', type=int, default=[1], nargs='+', help='list of numbers of filters.')
parser.add_argument('--sks', type=int, default=[1], nargs='+', help='list of sizes of kernels.')
parser.add_argument('--nz', type=int, default=128, help='size of the latent z vector. default=128.')
parser.add_argument('--input_size', type=int, default=[30, 28, 1], nargs='+', help='input shape. default=[30, 36, 1].')
parser.add_argument('--denoising', action='store_true', default=False, help="training with denoising. need 'dirty_sample' folder ins 'data_path'")
parser.add_argument('--dropout', action='store_true', default=False, help="do dropout'")
parser.add_argument('--latent_reg', action='store_true', default=False, help="do latent space regulization")
# path related ---------------------------------------------------------------
parser.add_argument('--data_path', type=str, default='/home/jm/etri_action_data/30_10', help='base path of dataset.')
parser.add_argument('--save_path', type=str, default='training_results', help='model save path.')
parser.add_argument('--tb_path', type=str, default='training_results', help="tensor board path. default='training_results'")
# ETC -------------------------------------------------------------------------
parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train for. default=100')
parser.add_argument('--batch_size', type=int, default=256, help='size of batch. default=256')
parser.add_argument('--save_period', type=int, default=50, help='network saving period. default=50')
parser.add_argument('--random_seed', type=int, help='manual random seed.')

options = parser.parse_args()
print(options)

# seed
if options.random_seed is None:
    options.random_seed = random.randint(1, 10000)

# check model arguments
assert(len(options.nfs) == len(options.sks))

# denoising
if options.denoising:
    options.dirty_sample_path = os.path.join(options.data_path, 'dirty_sample')
else:
    options.dirty_sample_path = None

if options.dropout:
    options.dropout = True
else:
    options.dropout = False

if options.latent_reg:
    options.latent_reg = True
else:
    options.latent_reg = False


# =============================================================================
# CALLBACKS
# =============================================================================
class BestLossCallBack(callbacks.Callback):
    def __init__(self, model, save_path, period=50, model_name=None):
        super(BestLossCallBack, self).__init__()
        self.model_to_save = model
        self.best_loss = 0
        self.save_path = save_path
        self.period = period
        self.counter = 0
        self.model_name = '' if model_name is None else '_' + model_name

    def on_epoch_end(self, epoch, logs=None):
        cur_loss = logs.get('loss')
        self.counter += 1
        if self.counter >= self.period:
            self.model_to_save.save(os.path.join(self.save_path, 'epoch_%04d%s.hdf5' % (epoch, self.model_name)))
            self.counter = 0
        if 0 == self.best_loss or self.best_loss > cur_loss:
            print('loss is improved')
            self.best_loss = cur_loss
            self.model_to_save.save(os.path.join(self.save_path, 'best_loss%s.hdf5' % self.model_name))

tbCallBack = callbacks.TensorBoard(log_dir=options.tb_path, histogram_freq=0, write_graph=True, write_images=True)
# mcCallBack = callbacks.ModelCheckpoint('./model/{epoch:05d}_epoch.hdf5',
#                                        monitor='mse', save_best_only=True, period=50)


# =============================================================================
# TRAINING
# =============================================================================
def train_network(opts):

    # load data
    action_info, action_data, target_data = load_samples(opts.data_path, options.dirty_sample_path)

    # construct model
    if 'AE' == opts.model:
        model, encoder, decoder = ConvAE(opts.nfs, opts.sks, opts.nz, opts.input_size, opts.dropout, opts.latent_reg)

    # elif 'VAE' == opts.model:
    #     model = ConvVAE(opts.nfs, opts.sks, opts.nz, opts.input_size)
    #     model.compile(optimizer='rmsprop', loss=vae_loss)
    #     model.fit(inputs=[action_data, target_data], epochs=opts.epochs, batch_size=opts.batch_size,
    #               callbacks=[tbCallBack, BestLossCallBack(model, opts.save_path, opts.save_period)], shuffle=True)
    else:
        print('There is no proper model for option: ' + opts.model)
        return

    callback_list = [tbCallBack,
                     BestLossCallBack(model, opts.save_path, opts.save_period)]
    # BestLossCallBack(decoder, opts.save_path, opts.save_period, 'decoder')

    # training
    model.compile(optimizer='rmsprop', loss='mse')
    model.fit(action_data, target_data, epochs=opts.epochs, batch_size=opts.batch_size,
              callbacks=callback_list, shuffle=True)

if __name__ == "__main__":

    options.timestamp = get_time_string()
    train_network(options)

# ()()
# ('') HAANJU.YOO
