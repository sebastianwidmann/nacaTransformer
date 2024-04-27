import json

import jax.debug
from absl import logging
from flax import traverse_util
from flax.training import train_state, orbax_utils
import orbax
from ml_collections import ConfigDict, FrozenConfigDict
import optax
import orbax.checkpoint as ocp
import tensorflow_datasets as tfds
import os
from typing import Any
import numpy as np
from jax.tree_util import tree_structure

from jax.experimental import mesh_utils
from jax.sharding import PositionalSharding

from src.transformer.input_pipeline import get_data_from_tfds
from src.transformer.network import VisionTransformer
from src.utilities.schedulers import load_learning_rate_scheduler
from src.utilities.visualisation import plot_delta, plot_loss, plot_fields
from src.utilities.pressure_preprocesing import *
from src.utilities.visualize_normalization_comparison import *

PRNGKey = Any


def create_train_state(params_key: PRNGKey, config: ConfigDict, lr_scheduler):

    if config.fine_tune.enable:

        if config.fine_tune.load_train_state:
            orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
            ckpt = orbax_checkpointer.restore(config.fine_tune.checkpoint_dir)
            return ckpt['model']

        model = VisionTransformer(config.vit)

        # Initialise model and use JIT to reside params in CPU memory
        orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        ckpt = orbax_checkpointer.restore(config.checkpoint_dir)
        restored_model = ckpt['model']
        variables = restored_model['params']

        # Initialise train state
        tx_trainable = optax.adamw(learning_rate=lr_scheduler,
                                   weight_decay=config.weight_decay)

        tx_frozen = optax.set_to_zero()

        partition_optimizers = {'trainable': tx_trainable, 'frozen': tx_frozen}

        trainable_layers = config.fine_tune.layers_to_train

        param_partitions = traverse_util.path_aware_map(
            lambda path, v: 'trainable' if any(layer in path for layer in trainable_layers) else 'trainable', variables)

        tx = optax.multi_transform(partition_optimizers, param_partitions)

        return train_state.TrainState.create(
            apply_fn=model.apply,
            params=variables,
            tx=tx
        )

    # if you want to continue training from a saved Train State
    if config.load_train_state:
        orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        ckpt = orbax_checkpointer.restore(config.checkpoint_dir)
        return ckpt['model']

    # Create model instance
    model = VisionTransformer(config.vit)

    # Initialise model and use JIT to reside params in GPU memory
    variables = jax.jit(lambda: model.init(params_key, jnp.ones(
        [config.batch_size, *config.vit.img_size, 1]), jnp.ones(
        [config.batch_size, *config.vit.img_size, 3]), train=False),
                        )()

    # Initialise train state
    tx = optax.adamw(learning_rate=lr_scheduler,
                     weight_decay=config.weight_decay)

    return train_state.TrainState.create(
        apply_fn=model.apply,
        params=variables['params'],
        tx=tx
    )

@jax.jit
def train_step(state: train_state.TrainState, x: jnp.ndarray, y: jnp.ndarray,
               key: PRNGKey):
    # Generate new dropout key for each step
    dropout_key = jax.random.fold_in(key, state.step)

    def loss_fn(params):
        preds = state.apply_fn({'params': params}, x, y, train=True,
                               rngs={'dropout': dropout_key},
                               )

        loss = optax.squared_error(preds, y).mean()

        return loss, preds

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, _), grads = grad_fn(state.params)

    state = state.apply_gradients(grads=grads)

    return state, loss

@jax.jit
def test_step(state: train_state.TrainState, x: jnp.ndarray, y: jnp.ndarray):
    preds = state.apply_fn({'params': state.params}, x, y, train=False)

    loss = optax.squared_error(preds, y).mean()

    return preds, loss


def train_and_evaluate(config: ConfigDict):

    if config.train_parallel:
        num_devices = len(jax.local_devices())
        devices = mesh_utils.create_device_mesh((num_devices,))
        sharding = PositionalSharding(devices).reshape(num_devices, 1, 1, 1)

    logging.info("Initialising dataset.")
    os.makedirs(config.output_dir, exist_ok=True)

    ds_train = get_data_from_tfds(config=config, mode='train')
    ds_test = get_data_from_tfds(config=config, mode='test')

    steps_per_epoch = ds_train.cardinality().numpy() / config.num_epochs
    total_steps = ds_train.cardinality().numpy()

    # Create PRNG key
    rng = jax.random.PRNGKey(0)
    rng_state = jax.random.PRNGKey(0)
    rng_idx = np.random.default_rng(0)
    sample_idx = rng_idx.integers(0, config.batch_size, 10)

    # Create learning rate scheduler
    lr_scheduler = load_learning_rate_scheduler(
        config=config, name=config.learning_rate_scheduler,
        total_steps=total_steps
    )

    state = create_train_state(rng_state, FrozenConfigDict(config), lr_scheduler)

    if config.train_parallel:
        state = jax.device_put(state, sharding.replicate())

    train_metrics, test_metrics, train_log, test_log = [], [], [], []

    logging.info("Starting training loop. Initial compile might take a while.")
    for step, batch in enumerate(tfds.as_numpy(ds_train)):

        if config.internal_geometry.set_internal_value:
            batch = set_geometry_internal_value(batch, config.internal_geometry.value)

        if config.pressure_preprocessing.enable:
            batch = pressure_preprocessing(batch, config)

        batch.pop('label')

        x_train = batch['encoder']
        y_train = batch['decoder']

        if config.train_parallel:
            x_train = jax.device_put(x_train,sharding)
            y_train = jax.device_put(y_train,sharding)

        state, train_loss = train_step(state, x_train, y_train, rng)
        train_log.append(train_loss)

        if (step + 1) % int(steps_per_epoch) == 0 and step != 0:
            epoch = int((step + 1) / int(steps_per_epoch))

            for test_batch in tfds.as_numpy(ds_test):

                if config.pressure_preprocessing.enable & config.visualisation.reverse_standardization:
                    mean_std = vectorized_get_mean_std(test_batch['decoder'],config.internal_geometry.value)

                if config.internal_geometry.set_internal_value:
                    test_batch = set_geometry_internal_value(batch,config.internal_geometry.value)

                if config.pressure_preprocessing.enable:
                    test_batch = pressure_preprocessing(test_batch, config)

                test_batch.pop('label')

                x_test = test_batch['encoder']
                y_test = test_batch['decoder']

                if config.train_parallel:
                    x_test = jax.device_put(x_test,sharding)
                    y_test = jax.device_put(y_test,sharding)

                predictions, test_loss = test_step(state, x_test, y_test)
                test_log.append(test_loss)

                if config.pressure_preprocessing.enable & config.visualisation.reverse_standardization:
                    y_test, predictions = reverse_standardize_batched(y_test, predictions, mean_std, config.internal_geometry.value)


            train_loss = np.mean(train_log)
            test_loss = np.mean(test_log)

            train_metrics.append(train_loss)
            test_metrics.append(test_loss)
            

            logging.info(
                'Epoch {}: Train_loss = {}, Test_loss = {}'.format(
                    epoch, train_loss, test_loss))

            # Reset epoch losses
            train_log.clear()
            test_log.clear()

            if epoch % config.output_frequency == 0:

                pickled_prediction_and_target = {'predictions':predictions, 'target':y_test , 'encoder_input': x_test}#pickle files in case you need values later to generate new plots
                save_pytree(pickled_prediction_and_target,'{}/pickled_files'.format(config.output_dir), 'epoch_'+str(epoch)+'.pckl')

                for i in sample_idx:
                    y_hat = predictions[i,:,:,:].squeeze()
                    y = y_test[i,:,:,:].squeeze()
                    x = x_test[i,:,:,:].squeeze()
                    plot_fields(config, y_hat, y, x,epoch,i)

                ckpt = {'model': state}
                orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
                save_args = orbax_utils.save_args_from_target(ckpt)
                orbax_checkpointer.save('{}/checkpoints/{}'.format(config.output_dir, str(epoch)), ckpt,
                                        save_args=save_args)

                    

            if epoch == config.num_epochs:

                pickled_prediction_and_target = {'predictions': predictions, 'target': y_test, 'encoder_input': x_test}#pickle files in case you need values later to generate new plots
                save_pytree(pickled_prediction_and_target, '{}/pickled_files'.format(config.output_dir), 'final.pckl')

                for i in sample_idx:
                    y_hat = predictions[i, :, :, :].squeeze()
                    y = y_test[i, :, :, :].squeeze()
                    x = x_test[i, :, :, :].squeeze()
                    plot_fields(config, y_hat, y, x, epoch, i)
                    plot_delta(config, y_hat, y, x, epoch, i)

    # Data analysis plots
    try:
        plot_loss(config, train_metrics, test_metrics)
    except ValueError:
        pass

    # save raw loss data into txt-file
    raw_loss = np.concatenate((train_metrics, test_metrics))
    raw_loss = raw_loss.reshape(2, -1).transpose()
    np.savetxt('{}/loss_raw.txt'.format(config.output_dir), raw_loss,
               delimiter=',')

    # write config file to check setup later if necessary
    config_dir ='{}/config'.format(config.output_dir)
    os.makedirs(config_dir, exist_ok=True)
    config_filepath = os.path.join(config_dir, 'config.txt')
    with open(config_filepath, "w") as outfile:
        config_dict = config.to_dict()
        for key, value in config_dict.items():
            outfile.write('%s:%s\n'%(key,value))

    ckpt = {'model': state}
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    save_args = orbax_utils.save_args_from_target(ckpt)
    orbax_checkpointer.save('{}/checkpoints/{}'.format(config.output_dir, 'Final'), ckpt,
                            save_args=save_args)

    

