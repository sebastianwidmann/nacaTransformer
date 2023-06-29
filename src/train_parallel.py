# ----------------------------------------------------------------------------
# Created By  : Sebastian Widmann
# Institution : TU Munich, Department of Aerospace and Geodesy
# Created Date: March 21, 2023
# version ='1.0'
# ---------------------------------------------------------------------------
from absl import logging
from flax import jax_utils
from flax.training import train_state, orbax_utils, common_utils
from flax.metrics import tensorboard
from functools import partial
import jax
import jax.numpy as jnp
from ml_collections import ConfigDict, FrozenConfigDict
import numpy as np
import optax
import orbax.checkpoint
import tensorflow_datasets as tfds

from typing import Any, Tuple

from src.transformer.input_pipeline import get_data_from_tfds
from src.transformer.network import VisionTransformer
from src.utilities.schedulers import load_learning_rate_scheduler
from src.utilities.visualisation import plot_predictions, plot_delta, \
    plot_loss, plot_fields

PRNGKey = Any


@partial(jax.pmap, static_broadcasted_argnums=(1, 2,))
def create_train_state(params_key: PRNGKey, config: ConfigDict, lr_scheduler):
    # Create model instance
    model = VisionTransformer(config.vit)

    # Initialise model and use JIT to reside params in CPU memory
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


@partial(jax.pmap, axis_name='num_devices')
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

    # Combine gradients and loss across devices
    loss = jax.lax.pmean(loss, axis_name='num_devices')
    grads = jax.lax.pmean(grads, axis_name='num_devices')

    # Synchronise state across devices with averaged gradient
    state = state.apply_gradients(grads=grads)

    return state, loss


@partial(jax.pmap, axis_name='num_devices')
def test_step(state: train_state.TrainState, x: jnp.ndarray, y: jnp.ndarray):
    preds = state.apply_fn({'params': state.params}, x, y, train=False)

    loss = optax.squared_error(preds, y).mean()
    loss = jax.lax.pmean(loss, axis_name='num_devices')

    return preds, loss


def train_and_evaluate(config: ConfigDict):
    # summary_writer = tensorboard.SummaryWriter(
    #     '/home/sebastianwidmann/Documents/git/nacaTransformer')
    # summary_writer.hparams(dict(config))

    logging.info("Initialising airfoilMNIST dataset.")

    ds_train = get_data_from_tfds(config=config, mode='train')
    ds_test = get_data_from_tfds(config=config, mode='test')

    steps_per_epoch = ds_train.cardinality().numpy() / config.num_epochs
    total_steps = ds_train.cardinality().numpy()

    # Create PRNG key
    rng = jax.random.PRNGKey(0)
    rng = jax.random.split(rng, jax.local_device_count())

    # Create learning rate scheduler
    lr_scheduler = load_learning_rate_scheduler(
        config=config, name=config.learning_rate_scheduler,
        total_steps=total_steps
    )

    # Create TrainState and map over GPUs
    state = create_train_state(rng, FrozenConfigDict(config), lr_scheduler)

    train_metrics, test_metrics, train_log, test_log = [], [], [], []

    logging.info("Starting training loop. Initial compile might take a while.")
    for step, batch in enumerate(tfds.as_numpy(ds_train)):
        x_train = common_utils.shard(batch['encoder'])
        y_train = common_utils.shard(batch['decoder'])

        state, train_loss = train_step(state, x_train, y_train, rng)
        train_log.append(train_loss)

        if (step + 1) % int(steps_per_epoch) == 0 and step != 0:
            epoch = int((step + 1) / int(steps_per_epoch))

            for test_batch in tfds.as_numpy(ds_test):
                x_test = common_utils.shard(test_batch['encoder'])
                y_test = common_utils.shard(test_batch['decoder'])

                preds, test_loss = test_step(state, x_test, y_test)
                test_log.append(test_loss)

            train_loss = np.mean(train_log)
            test_loss = np.mean(test_log)

            logging.info(
                'Epoch {}: Train_loss = {}, Test_loss = {}'.format(
                    epoch, train_loss, test_loss))

            # Reset epoch losses
            train_log.clear()
            test_log.clear()

            # summary_writer.scalar('train_loss', train_loss, epoch)
            # summary_writer.scalar('test_loss', tess_loss, epoch)

            # TODO: fix indexing for sharded data [num_devices, batch_size, ...]
            if epoch % config.output_frequency == 0:
                pred_data = preds[::, 0].squeeze()
                test_data = y_test[::, 0].squeeze()
                plot_predictions(config, pred_data, test_data, epoch)

            if epoch == config.num_epochs:
                pred_data = preds[::, 0].squeeze()
                test_data = y_test[::, 0].squeeze()
                plot_delta(config, pred_data, test_data, epoch, 'cividis')
                plot_fields(config, pred_data, test_data, epoch)

    # summary_writer.flush()

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

    # Save model
    ckpt = {'model': state}
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    save_args = orbax_utils.save_args_from_target(ckpt)
    orbax_checkpointer.save('{}/nacaVIT'.format(config.output_dir), ckpt,
                            save_args=save_args)
