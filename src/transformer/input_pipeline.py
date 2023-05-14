# ----------------------------------------------------------------------------
# Created By  : Sebastian Widmann
# Institution : TU Munich, Department of Aerospace and Geodesy
# Created Date: April 9, 2023
# version ='1.0'
# ---------------------------------------------------------------------------

import tensorflow_datasets as tfds
import tensorflow as tf


def get_data_from_tfds(*, config, mode, num_prefetch=1):
    builder = tfds.builder_from_directory(builder_dir=config.readdir)

    dataset = builder.as_dataset(
        split=tfds.split_for_jax_process(mode),
        shuffle_files=True,
        # batch_size=config.batch_size,
    )

    dataset = dataset.map(
        lambda sample: {
            'encoder': sample['data_encoder'],
            'decoder': sample['data_decoder'],
        })

    if mode == 'train':
        dataset = dataset.repeat(config.num_epochs).shuffle(
            buffer_size=1000,
            reshuffle_each_iteration=True,
        )
        dataset = dataset.batch(batch_size=config.batch_size,
                                num_parallel_calls=tf.data.AUTOTUNE).prefetch(
            num_prefetch)
    elif mode == 'test':
        dataset = dataset.shuffle(
            buffer_size=1000,
            reshuffle_each_iteration=True,
        )
        dataset = dataset.batch(batch_size=config.batch_size,
                                num_parallel_calls=tf.data.AUTOTUNE).prefetch(
            num_prefetch)

    return dataset
