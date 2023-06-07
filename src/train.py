# ----------------------------------------------------------------------------
# Created By  : Sebastian Widmann
# Institution : TU Munich, Department of Aerospace and Geodesy
# Created Date: March 21, 2023
# version ='1.0'
# ---------------------------------------------------------------------------
from absl import logging
from flax.training import train_state, orbax_utils
import functools
import jax
import jax.numpy as jnp
from ml_collections import ConfigDict
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


# @functools.partial(jax.pmap, static_broadcasted_argnums=0)
def create_train_state(config: ConfigDict, lr_scheduler, rng: PRNGKey) -> \
        train_state.TrainState:
    # Create model instance
    model = VisionTransformer(config.vit)

    # Initialise model and use JIT to reside params in CPU memory
    variables = jax.jit(lambda: model.init(rng, jnp.ones(
        [config.batch_size, *config.vit.img_size, 1]), jnp.ones(
        [config.batch_size, *config.vit.img_size, 3]), train=False),
                        )()

    # Initialise train state
    tx = optax.adamw(learning_rate=lr_scheduler,
                     weight_decay=config.weight_decay)

    return train_state.TrainState.create(
        apply_fn=model.apply, params=variables['params'], tx=tx)


@jax.jit
# @functools.partial(jax.pmap, axis_name='ensemble')
def train_step(state: train_state.TrainState, batch: jnp.ndarray,
               rng: PRNGKey) -> Tuple[train_state.TrainState, Any]:
    # Generate new dropout key for each step
    rng_dropout = jax.random.fold_in(rng, state.step)

    def loss_fn(params):
        preds = state.apply_fn({'params': params},
                               batch['encoder'], batch['decoder'],
                               train=True,
                               rngs={'dropout': rng_dropout},
                               )

        loss = optax.squared_error(preds, batch['decoder']).mean()

        return loss, preds

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, preds), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)

    return state, loss


@jax.jit
# @jax.pmap
def test_step(state: train_state.TrainState, batch: jnp.ndarray):
    preds = state.apply_fn({'params': state.params},
                           batch['encoder'], batch['decoder'],
                           train=False,
                           )

    loss = optax.squared_error(preds, batch['decoder']).mean()

    return preds, loss


def train_and_evaluate(config: ConfigDict):
    logging.info("Initialising airfoilMNIST dataset.")

    ds_train = get_data_from_tfds(config=config, mode='train')
    ds_test = get_data_from_tfds(config=config, mode='test')

    steps_per_epoch = ds_train.cardinality().numpy() / config.num_epochs
    total_steps = ds_train.cardinality().numpy()

    # Create PRNG key
    rng = jax.random.PRNGKey(0)
    # Split PRNG key into required keys
    rng, rng_params, rng_dropout = jax.random.split(rng, num=3)

    # Create learning rate scheduler
    lr_scheduler = load_learning_rate_scheduler(
        name=config.learning_rate_scheduler,
        total_steps=total_steps,
    )

    # Create TrainState
    state = create_train_state(config, lr_scheduler, rng_params)
    # state = create_train_state(config, jax.random.split(rng_params,
    #                                                     jax.device_count()))

    train_metrics, test_metrics, train_log, test_log = [], [], [], []

    logging.info("Starting training loop. Initial compile might take a while.")

    for step, batch in enumerate(tfds.as_numpy(ds_train)):
        state, train_loss = train_step(state, batch, rng_dropout)
        # state, train_loss = pad_shard_unpad(train_step)(state,
        #                                                 batch,
        #                                                 rng_dropout,
        #                                                 min_device_batch=config.batch_size)
        train_log.append(train_loss)

        if (step + 1) % int(steps_per_epoch) == 0 and step != 0:
            epoch = int((step + 1) / int(steps_per_epoch))

            for test_batch in tfds.as_numpy(ds_test):
                preds, test_loss = test_step(state, test_batch)
                test_log.append(test_loss)

            train_loss = np.mean(train_log)
            test_loss = np.mean(test_log)

            train_metrics.append(train_loss)
            test_metrics.append(test_loss)

            logging.info(
                'Epoch {}: Train_loss = {:.6f}, Test_loss = {:.6f}'.format(
                    epoch, train_loss, test_loss))

            if epoch % config.output_frequency == 0:
                plot_predictions(config, preds[0, :, :, ],
                                 test_batch['decoder'][0, :, :], epoch)

            if epoch == config.num_epochs:
                plot_delta(config, preds[0, :, :, ], test_batch['decoder'][0,
                                                     :, :, ], epoch, 'cividis')
                plot_fields(config, preds[0, :, :, ], test_batch['decoder'][0,
                                                      :, :, ], epoch)

    # Data analysis plots
    try:
        plot_loss(config, train_metrics, test_metrics)
    except ValueError:
        pass

    # save raw loss data into txt-file
    raw_loss = np.concatenate((train_metrics, test_metrics))
    raw_loss = raw_loss.reshape(2, -1).transpose()
    np.savetxt('loss_raw.txt', raw_loss, delimiter=',')

    # Save model
    ckpt = {'model': state}
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    save_args = orbax_utils.save_args_from_target(ckpt)
    orbax_checkpointer.save('nacaVIT', ckpt, save_args=save_args)
