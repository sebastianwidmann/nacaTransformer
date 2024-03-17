# ----------------------------------------------------------------------------
# Created By  : Sebastian Widmann
# Institution : TU Munich, Department of Aerospace and Geodesy
# Created Date: April 9, 2023
# version ='1.0'
# ---------------------------------------------------------------------------
import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np


def get_data_from_tfds(*, config, mode):
    builder = tfds.builder_from_directory(builder_dir=config.fine_tune_dataset)

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


def calculate_mean_std(*, config):
    builder = tfds.builder_from_directory(builder_dir=config.fine_tune_dataset)
    ds_train = builder.as_dataset(split=tfds.split_for_jax_process('train'))
    ds_test = builder.as_dataset(split=tfds.split_for_jax_process('test'))

    sample_means, sample_vars = [], []

    for batch in tfds.as_numpy(ds_train):
        p = batch['decoder'][:, :, 0]
        ux = batch['decoder'][:, :, 1]
        uy = batch['decoder'][:, :, 2]

        sample_means.append([p.mean(), ux.mean(), uy.mean()])
        sample_vars.append([p.var(), ux.var(), uy.var()])

    for batch in tfds.as_numpy(ds_test):
        p = batch['decoder'][:, :, 0]
        ux = batch['decoder'][:, :, 1]
        uy = batch['decoder'][:, :, 2]

        sample_means.append([p.mean(), ux.mean(), uy.mean()])
        sample_vars.append([p.var(), ux.var(), uy.var()])

    total_mean = np.average(sample_means, axis=0)
    total_std = np.sqrt(np.average(sample_vars, axis=0))
    return total_mean, total_std


def standardise_dataset(batch, mean, std):
    for i in range(batch['decoder'].shape[-1]):
        batch['decoder'][:, :, :, i] = (batch['decoder'][:, :, :,
                                        i] - mean[i]) / std[i]
    return batch
