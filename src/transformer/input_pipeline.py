# ----------------------------------------------------------------------------
# Created By  : Sebastian Widmann
# Institution : TU Munich, Department of Aerospace and Geodesy
# Created Date: April 9, 2023
# version ='1.0'
# ---------------------------------------------------------------------------

import tensorflow_datasets as tfds
import tensorflow as tf


def get_data_from_tfds(*, config, mode):
    builder = tfds.builder_from_directory(builder_dir=config.dataset)

    dataset = builder.as_dataset(
        split=tfds.split_for_jax_process(mode),
        shuffle_files=True,
    )

    if mode == 'train':
        dataset = dataset.shuffle(config.shuffle_buffer_size,
                                  seed=0,
                                  reshuffle_each_iteration=True)

        dataset = dataset.repeat(config.num_epochs)

    dataset = dataset.batch(batch_size=config.batch_size,
                            drop_remainder=True,
                            num_parallel_calls=tf.data.AUTOTUNE).prefetch(
        tf.data.AUTOTUNE)

    return dataset
