from absl import app
from absl import flags
from ml_collections import config_flags
import os
import random
import tensorflow as tf
from tqdm import tqdm

from conversion import vtk_to_tfTensor, create_tfExample

# from conversion import generate_tfrecords

FLAGS = flags.FLAGS

config_flags.DEFINE_config_file('config')


def main(argv):
    """
    Convert vtk datasets from airfoilMNIST (readdir) into the TFRecord format
    and save them as into the TFRecords directory (writedir). More
    information about TFRecords can be found on
    https://www.tensorflow.org/tutorials/load_data/tfrecord.
    """
    # parse nested config for preprocessing variables
    config = FLAGS.config.preprocess

    # check if output directory exists and create dir if necessary
    if not os.path.exists(config.writedir):
        os.makedirs(config.writedir)

    vtu_folder = os.path.join(config.readdir, 'vtu')
    stl_folder = os.path.join(config.readdir, 'stl')

    vtu_list = [x for x in sorted(os.listdir(vtu_folder)) if x.endswith('.vtu')]
    stl_list = [("_".join(x.split("_", 2)[:2]) + ".stl") for x in vtu_list]

    dataset = list(zip(vtu_list, stl_list))

    quotient, remainder = divmod(len(dataset), config.nsamples)
    n_tfrecords = quotient + remainder

    for i in tqdm(range(n_tfrecords), desc='TFRecords'):
        if remainder != 0 and i == n_tfrecords - 1:
            samples = [dataset.pop(random.randrange(len(dataset))) for _ in
                       range(remainder)]
        else:
            samples = [dataset.pop(random.randrange(len(dataset))) for _ in
                       range(config.nsamples)]

        file_dir = os.path.join(config.writedir,
                                'airfoilMNIST_{}.tfrecord'.format(i))

        with tf.io.TFRecordWriter(file_dir) as writer:
            for sample in samples:
                vtu_dir = os.path.join(vtu_folder, sample[0])
                stl_dir = os.path.join(stl_folder, sample[1])

                # split string to extract information about airfoil, angle of
                # attack and Mach number to write as feature into tfrecord
                sim_config = sample[0].rsplit('.', 1)[0]
                sim_config = sim_config.split('_')

                airfoil, angle, mach = sim_config
                angle, mach = float(angle), float(mach)

                data = vtk_to_tfTensor(vtu_dir, stl_dir, config)

                example = create_tfExample(airfoil, angle, mach, data)

                writer.write(example.SerializeToString())


if __name__ == '__main__':
    app.run(main)
