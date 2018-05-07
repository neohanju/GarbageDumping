import argparse
import os
from keras.models import load_model, Model
from utils import load_samples, save_samples, save_latent_variables, load_latent_vectors

# =============================================================================
# OPTIONS
# =============================================================================
parser = argparse.ArgumentParser(description='Test and generate reconstructions.')
# path related ---------------------------------------------------------------
parser.add_argument('--model_path', type=str, default='training_results/0000-00-00_00-00-00/best_loss.hdf5', help="trained model path.")
parser.add_argument('--data_path', type=str, default='training_results/0000-00-00_00-00-00/latents', help='base path of dataset.')
parser.add_argument('--save_path', type=str, default='training_results/0000-00-00_00-00-00', help='model save path.')

options = parser.parse_args()
print(options)


def validate_decoder(target_path, generation_path):
    from numpy import linalg as LA
    import numpy as np

    target_info, targets, _ = load_samples(target_path)
    generate_info, generates, _ = load_samples(generation_path)

    if len(targets) != len(generates):
        return False

    for i in range(len(targets)):
        if LA.norm(np.subtract(targets[i], generates[i])) > 0.001:
            return False
    return True


if __name__ == "__main__":
    test_info, test_data = load_latent_vectors(options.data_path)
    model = load_model(options.model_path)

    decoder = Model(inputs=model.get_layer('sequential_2').model.input,
                    outputs=model.get_layer('sequential_2').model.output)
    generations = decoder.predict(test_data)
    save_samples(os.path.join(options.save_path, 'generates'), generations, test_info)

    print(validate_decoder(os.path.join(options.save_path, 'recons'),
                           os.path.join(options.save_path, 'generates')))
