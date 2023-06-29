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

    ds = builder.as_dataset(
        split=tfds.split_for_jax_process(mode),
    )

    if mode == 'train':
        ds = ds.shuffle(config.shuffle_buffer_size,
                        seed=0,
                        reshuffle_each_iteration=True)

        ds = ds.repeat(config.num_epochs)

    ds = ds.batch(batch_size=config.batch_size,
                  drop_remainder=True,
                  num_parallel_calls=tf.data.AUTOTUNE)

    return ds.prefetch(tf.data.AUTOTUNE)
