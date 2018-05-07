import argparse
import os
from keras.models import load_model, Model
from utils import load_samples, save_samples, save_latent_variables, save_recon_error

# =============================================================================
# OPTIONS
# =============================================================================
parser = argparse.ArgumentParser(description='Test and generate reconstructions.')
# path related ---------------------------------------------------------------
parser.add_argument('--model_path', type=str, default='training_results/0000-00-00_00-00-00/best_loss.hdf5', help="trained model path.")
parser.add_argument('--data_path', type=str, default='/home/mlpa/Workspace/dataset/etri_action_data/30_10/posetrack', help='base path of dataset.')
parser.add_argument('--save_path', type=str, default='training_results/0000-00-00_00-00-00', help='model save path.')
parser.add_argument('--save_latent', action='store_true', default=False, help='save latent variables')

options = parser.parse_args()
print(options)


if __name__ == "__main__":
    test_info, test_data, _ = load_samples(options.data_path)
    model = load_model(options.model_path)
    predictions = model.predict(test_data)
    save_samples(os.path.join(options.save_path, 'recons'), predictions, test_info)

    loss_and_metrics = model.evaluate(test_data, predictions)
    print(loss_and_metrics)
    save_recon_error(os.path.join((options.save_path, 'recon_errors')), loss_and_metrics, test_info)

    if options.save_latent:
        encoder = Model(inputs=model.get_layer('sequential_1').model.input,
                        outputs=model.get_layer('sequential_1').model.output)
        latents = encoder.predict(test_data)
        save_latent_variables(os.path.join(options.save_path, 'latents'), latents, test_info)


# ()()
# ('') HAANJU.YOO
