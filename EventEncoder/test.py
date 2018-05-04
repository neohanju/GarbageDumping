import argparse
import os
from keras.models import load_model, Model
from utils import load_samples, save_samples

# =============================================================================
# OPTIONS
# =============================================================================
parser = argparse.ArgumentParser(description='Test and generate reconstructions.')
# path related ---------------------------------------------------------------
parser.add_argument('--model_path', type=str, default='training_results', help="trained model path.")
parser.add_argument('--data_path', type=str, default='/home/jm/etri_action_data/30_10', help='base path of dataset.')
parser.add_argument('--save_path', type=str, default='training_results/recons', help='model save path.')
parser.add_argument('--save_latent', action='store_true', default=False, help='save latent variables')

options = parser.parse_args()
print(options)


if __name__ == "__main__":
    test_data, _, file_names = load_samples(options.data_path)
    model = load_model(options.model_path)
    predictions = model.predict(test_data)
    save_samples(options.save_path, predictions, file_names)

    if options.save_latent:
        save_path = os.path.join(os.path.dirname(options.save_path), 'latents')
        latent_file_names = []
        for file_name in file_names:
            latent_file_names.append(file_name.split('-recon.npy')[0] + '-latent.npy')

        encoder = Model(inputs=model.input, outputs=model.get_layer('latent').output)
        latents = encoder.predict(test_data)
        save_samples(options.save_path, latents, latent_file_names)

# ()()
# ('') HAANJU.YOO
