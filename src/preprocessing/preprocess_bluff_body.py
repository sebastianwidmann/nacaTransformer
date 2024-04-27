from ml_collections import ConfigDict
import os
import random

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # disable INFO and WARNING messages
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
from tqdm import tqdm

from conversion_bluff_body import vtk_to_tfTensor, create_tfExample

def generate_tfds_dataset(config: ConfigDict):
    """
    Convert vtk datasets from bluff body (readdir) into the TFRecord format
    and save them as into the TFRecords directory (writedir). More
    information about TFRecords can be found on
    https://www.tensorflow.org/tutorials/load_data/tfrecord.
    """

    # check if output directory exists and create dir if necessary
    if not os.path.exists(config.preprocess.writedir):
        os.makedirs(config.preprocess.writedir)

    vtu_folder = os.path.join(config.preprocess.readdir, 'vtu')
    stl_folder = os.path.join(config.preprocess.readdir, 'stl')

    vtu_list = os.listdir(vtu_folder)
    vtu_list = sorted(vtu_list)
    stl_list = os.listdir(stl_folder)
    stl_list = sorted(stl_list)

    dataset = list(zip(vtu_list, stl_list))
   

    ds_train = [dataset.pop(random.randrange(len(dataset))) for _ in
                range(int(config.preprocess.train_split * len(dataset)))]
    ds_test = dataset

    train_quotient, train_remainder = divmod(len(ds_train),
                                             config.preprocess.nsamples)
    test_quotient, test_remainder = divmod(len(ds_test),
                                           config.preprocess.nsamples)

    n_files_train = (train_quotient + 1 if train_remainder != 0 else
                     train_quotient)
    n_files_test = (test_quotient + 1 if test_remainder != 0 else test_quotient)

    train_shards, test_shards = [], []

    for i in tqdm(range(n_files_train), desc='Train split', position=0):
        if train_remainder != 0 and i == n_files_train - 1:
            batch = [ds_train.pop(random.randrange(len(ds_train))) for _ in
                     range(train_remainder)]
        else:
            batch = [ds_train.pop(random.randrange(len(ds_train))) for _ in
                     range(config.preprocess.nsamples)]

        file_dir = os.path.join(config.preprocess.writedir,
                                'bluff_body-train.tfrecord-{}-of-{}'.format(
                                    str(i).zfill(5),
                                    str(n_files_train).zfill(
                                        5)))

        with tf.io.TFRecordWriter(file_dir) as writer:
            j = 0
            for sample in tqdm(batch, desc='Shards', position=1,
                               leave=False):
                # split string to extract information about airfoil, angle of
                # attack and Mach number to write as feature into tfrecord
                sim_config = sample[0][:-4]

                bluff_body, mach = sim_config.split('_')
                mach = float(mach)

                vtu_dir = os.path.join(vtu_folder, sample[0])
                stl_dir = os.path.join(stl_folder, sample[1])
                try:
                    x, y = vtk_to_tfTensor(config, vtu_dir, stl_dir, mach)
                    example = create_tfExample(x, y, sim_config)
                    writer.write(example.SerializeToString())

                    j += 1
                except ValueError:
                    print('train, ValueError: ', vtu_dir)
                    continue
                except AttributeError:
                    print('train, Attribute error: ', vtu_dir)
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
                                'bluff_body-test.tfrecord-{}-of-{}'.format(
                                    str(i).zfill(5),
                                    str(n_files_test).zfill(
                                        5)))

        with tf.io.TFRecordWriter(file_dir) as writer:
            j = 0
            for sample in tqdm(batch, desc='Shards', position=1,
                               leave=False):
                # split string to extract information about airfoil, angle of
                # attack and Mach number to write as feature into tfrecord
                sim_config = sample[0][:-4]

                bluff_body, mach = sim_config.split('_')
                mach = float(mach)

                vtu_dir = os.path.join(vtu_folder, sample[0])
                stl_dir = os.path.join(stl_folder, sample[1])

                try:
                    x, y = vtk_to_tfTensor(config, vtu_dir, stl_dir, mach)
                    example = create_tfExample(x, y, sim_config)
                    writer.write(example.SerializeToString())

                    j += 1
                except ValueError:
                    print(vtu_dir)
                    continue
                except AttributeError:
                    print(vtu_dir)
                    continue

        test_shards.append(j)

    # Create metadata files to read dataset with tfds.load(args)
    features = tfds.features.FeaturesDict({
        'encoder': tfds.features.Tensor(
            shape=(*config.vit.img_size, 1),
            dtype=np.float32,
        ),
        'decoder': tfds.features.Tensor(
            shape=(*config.vit.img_size, 3),
            dtype=np.float32,
        ),
        'label': tfds.features.Text(
            encoder=None,
            encoder_config=None,
            doc='Simulation config: bluff_body_mach'
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
