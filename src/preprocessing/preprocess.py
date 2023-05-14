# ----------------------------------------------------------------------------
# Created By  : Sebastian Widmann
# Institution : TU Munich, Department of Aerospace and Geodesy
# Created Date: April 9, 2023
# version ='1.0'
# ---------------------------------------------------------------------------
from absl import app
from absl import flags
from ml_collections import config_flags
import os
import random
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
from tqdm import tqdm
from conversion import vtk_to_tfTensor, create_tfExample

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file(
    'config',
    'config.py',
    'File path to hyperparameter configuration',
    lock_config=True,
)
flags.mark_flag_as_required('config')


def main(argv):
    """
    Convert vtk datasets from airfoilMNIST (readdir) into the TFRecord format
    and save them as into the TFRecords directory (writedir). More
    information about TFRecords can be found on
    https://www.tensorflow.org/tutorials/load_data/tfrecord.
    """

    config = FLAGS.config

    # check if output directory exists and create dir if necessary
    if not os.path.exists(config.preprocess.writedir):
        os.makedirs(config.preprocess.writedir)

    vtu_folder = os.path.join(config.preprocess.readdir, 'vtu')
    stl_folder = os.path.join(config.preprocess.readdir, 'stl')

    vtu_list = [x for x in sorted(os.listdir(vtu_folder)) if x.endswith('.vtu')]
    stl_list = [("_".join(x.split("_", 2)[:2]) + ".stl") for x in vtu_list]

    dataset = list(zip(vtu_list, stl_list))

    ds_train = [dataset.pop(random.randrange(len(dataset))) for _ in
                range(int(config.train_size * len(dataset)))]
    ds_test = dataset

    train_quotient, train_remainder = divmod(len(ds_train),
                                             config.preprocess.nsamples)
    test_quotient, test_remainder = divmod(len(ds_test),
                                           config.preprocess.nsamples)

    n_files_train = (train_quotient + 1 if train_remainder != 0 else
                     train_quotient)
    n_files_test = (test_quotient + 1 if test_remainder != 0 else test_quotient)

    train_shards, test_shards = [], []

    with open('errors.txt', 'w') as f:
        for i in tqdm(range(n_files_train), desc='Train split', position=0):
            if train_remainder != 0 and i == n_files_train - 1:
                batch = [ds_train.pop(random.randrange(len(ds_train))) for _ in
                         range(train_remainder)]
            else:
                batch = [ds_train.pop(random.randrange(len(ds_train))) for _ in
                         range(config.preprocess.nsamples)]

            file_dir = os.path.join(config.preprocess.writedir,
                                    'airfoilMNIST-train.tfrecord-{}-of-{}'.format(
                                        str(i).zfill(5),
                                        str(n_files_train).zfill(
                                            5)))

            with tf.io.TFRecordWriter(file_dir) as writer:
                j = 0
                for sample in tqdm(batch, desc='Shards', position=1,
                                   leave=False):
                    # split string to extract information about airfoil, angle of
                    # attack and Mach number to write as feature into tfrecord
                    sim_config = sample[0].rsplit('.', 1)[0]
                    sim_config = sim_config.split('_')

                    airfoil, angle, mach = sim_config
                    angle, mach = float(angle), float(mach)

                    vtu_dir = os.path.join(vtu_folder, sample[0])
                    stl_dir = os.path.join(stl_folder, sample[1])
                    try:
                        x, y = vtk_to_tfTensor(vtu_dir, stl_dir,
                                               config.preprocess, mach)
                        example = create_tfExample(x, y)

                        writer.write(example.SerializeToString())

                        j += 1
                    except ValueError:
                        print('train, ValueError: ', vtu_dir, file=f)
                        continue
                    except AttributeError:
                        print('train, Attribute error: ', vtu_dir, file=f)
                        continue

            train_shards.append(j)

        for i in tqdm(range(n_files_test), desc='Test split', position=0):
            if test_remainder != 0 and i == n_files_test - 1:
                batch = [ds_test.pop(random.randrange(len(ds_test))) for _ in
                         range(test_remainder)]
            else:
                batch = [ds_test.pop(random.randrange(len(ds_test))) for _ in
                         range(config.preprocess.nsamples)]

            file_dir = os.path.join(config.preprocess.writedir,
                                    'airfoilMNIST-test.tfrecord-{}-of-{}'.format(
                                        str(i).zfill(5),
                                        str(n_files_test).zfill(
                                            5)))

            with tf.io.TFRecordWriter(file_dir) as writer:
                j = 0
                for sample in tqdm(batch, desc='Shards', position=1,
                                   leave=False):
                    # split string to extract information about airfoil, angle of
                    # attack and Mach number to write as feature into tfrecord
                    sim_config = sample[0].rsplit('.', 1)[0]
                    sim_config = sim_config.split('_')

                    airfoil, angle, mach = sim_config
                    angle, mach = float(angle), float(mach)

                    vtu_dir = os.path.join(vtu_folder, sample[0])
                    stl_dir = os.path.join(stl_folder, sample[1])

                    try:
                        x, y = vtk_to_tfTensor(vtu_dir, stl_dir,
                                               config.preprocess, mach)
                        example = create_tfExample(x, y)

                        writer.write(example.SerializeToString())

                        j += 1
                    except ValueError:
                        print(vtu_dir, file=f)
                        continue
                    except AttributeError:
                        print(vtu_dir, file=f)
                        continue

            test_shards.append(j)

    f.close()

    # Create metadata files to read dataset with tfds.load(args)
    features = tfds.features.FeaturesDict({
        'data_encoder': tfds.features.Tensor(
            shape=(*config.preprocess.resolution, 2),
            dtype=np.float32,
        ),
        'data_decoder': tfds.features.Tensor(
            shape=(*config.preprocess.resolution, 3),
            dtype=np.float32,
        ),
    })

    split_infos = [
        tfds.core.SplitInfo(
            name='train',
            shard_lengths=train_shards,
            num_bytes=0,
        ),
        tfds.core.SplitInfo(
            name='test',
            shard_lengths=test_shards,
            num_bytes=0,
        ),
    ]

    tfds.folder_dataset.write_metadata(
        data_dir=config.preprocess.writedir,
        features=features,
        split_infos=split_infos,
        filename_template='{DATASET}-{SPLIT}.{FILEFORMAT}-{SHARD_X_OF_Y}',
    )


if __name__ == '__main__':
    app.run(main)
