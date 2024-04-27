import os
import numpy as np
import pickle
from naca_transformer.code.src.utilities.visualisation import plot_fields_preprocess_comparison
from naca_transformer.code.config import get_config


#TODO clean up code in this file and maybe make it run from a config file

def save_pytree(pytree, directory, filename):
    os.makedirs(directory, exist_ok=True)
    filepath = os.path.join(directory, filename)
    with open(filepath, 'wb') as f:
        pickle.dump(pytree, f)


def load_pytree(filename):
    with open(filename, 'rb') as f:
        pytree = pickle.load(f)
    return pytree


def make_comparison_plots(normalized_training, not_normalized_training,indexes, output_dir ,epoch):
    # TODO have to load the saved config file from
    config = get_config()

    normalized_pytree = load_pytree(normalized_training)
    not_normalized_pytree = load_pytree(not_normalized_training)

    predictions_normalized = normalized_pytree['predictions']
    predictions_not_normalized = not_normalized_pytree['predictions']
    ground_truth_not_normalized = not_normalized_pytree['target']
    encoder_input = not_normalized_pytree['encoder_input']

    b, h, w, c = predictions_normalized.shape

    for i in indexes:
        plot_fields_preprocess_comparison(config, output_dir,predictions_not_normalized[i, :, :, :],
                                          predictions_normalized[i, :, :, :],
                                          ground_truth_not_normalized[i, :, :, :], encoder_input[i, :, :, :], epoch, i)


if __name__ == '__main__':

    normalized_file_directory = '/local/disk1/ebeqa/naca_transformer/Outputs/bluff_fine_tuning_normalized/pickled_files'
    not_normalized_file_directory = '/local/disk1/ebeqa/naca_transformer/Outputs/bluff_fine_tuning/pickled_files'
    output_dir = '/local/disk1/ebeqa/naca_transformer/Outputs/bluff_normalization_comparison'

    os.makedirs(output_dir, exist_ok=True)

    normalized_files = os.listdir(normalized_file_directory)
    not_normalized_files = os.listdir(not_normalized_file_directory)

    normalized_files.sort()
    not_normalized_files.sort()

    num_normalized_files = len(normalized_files)
    num_not_normalized_files = len(not_normalized_files)

    indexes = np.random.randint(0, 60, 10)

    # TODO have to also check if epochs are the same in files
    if (num_normalized_files == num_not_normalized_files):
        for i, filename in enumerate(not_normalized_files):
            make_comparison_plots(os.path.join(normalized_file_directory, filename),
                                  os.path.join(not_normalized_file_directory, filename), indexes,output_dir,5 * (i + 1))
