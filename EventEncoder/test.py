import argparse
import os
from keras.models import load_model
from utils import load_samples, save_samples

# =============================================================================
# OPTIONS
# =============================================================================
parser = argparse.ArgumentParser(description='Test and generate reconstructions.')
# path related ---------------------------------------------------------------
parser.add_argument('--model_path', type=str, default='training_results', help="trained model path.")
parser.add_argument('--data_path', type=str, default='/home/jm/etri_action_data/30_10', help='base path of dataset.')
parser.add_argument('--save_path', type=str, default='training_results/recons', help='model save path.')

options = parser.parse_args()
print(options)

if __name__ == "__main__":
    test_data, _, file_names = load_samples(options.data_path)
    model = load_model(options.model_path)
    predictions = model.predict(test_data)
    save_samples(options.save_path, predictions, file_names)

# ()()
# ('') HAANJU.YOO
