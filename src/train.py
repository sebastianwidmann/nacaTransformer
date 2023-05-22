# ----------------------------------------------------------------------------
# Created By  : Sebastian Widmann
# Institution : TU Munich, Department of Aerospace and Geodesy
# Created Date: March 21, 2023
# version ='1.0'
# ---------------------------------------------------------------------------
from absl import logging
from flax.training import train_state, orbax_utils
import jax
import jax.numpy as jnp
from ml_collections import ConfigDict
import numpy as np
import optax
import orbax.checkpoint
import tensorflow_datasets as tfds

from typing import Any, Tuple

from src.transformer.input_pipeline import get_data_from_tfds
from src.transformer.vit import VisionTransformer
from src.utilities.visualisation import plot_prediction, plot_loss

PRNGKey = Any


def create_train_state(config: ConfigDict, model: VisionTransformer,
                       rng: PRNGKey) -> train_state.TrainState:
    # Initialise model and use JIT to reside params in CPU memory
    variables = jax.jit(lambda: model.init(rng, jnp.ones(
        [config.batch_size, *config.vit.img_size, 1]), jnp.ones(
        [config.batch_size, *config.vit.img_size, 3]), train=False),
                        backend='cpu')()

    # Initialise train state
    tx = optax.adamw(learning_rate=config.learning_rate,
                     weight_decay=config.weight_decay)

    return train_state.TrainState.create(
        apply_fn=model.apply, params=variables['params'], tx=tx)


@jax.jit
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
def test_step(state: train_state.TrainState, batch: jnp.ndarray):
    preds = state.apply_fn({'params': state.params},
                           batch['encoder'], batch['decoder'],
                           train=False,
                           )

    loss = optax.squared_error(preds, batch['decoder']).mean()

    return preds, loss


def learning_rate_scheduler(num_steps, warmup_steps):
    return num_steps


def train_and_evaluate(config: ConfigDict):
    logging.info("Initialising airfoilMNIST dataset.")

    ds_train = get_data_from_tfds(config=config, mode='train')
    ds_test = get_data_from_tfds(config=config, mode='test')

    # Create PRNG key
    rng = jax.random.PRNGKey(0)
    # Split PRNG key into required keys
    rng, rng_params, rng_dropout = jax.random.split(rng, num=3)

    # Create model instance and TrainState
    model = VisionTransformer(config.vit)
    state = create_train_state(config, model, rng_params)

    steps_per_epoch = ds_train.cardinality().numpy() / config.num_epochs

    train_metrics, test_metrics, train_log, test_log = [], [], [], []

    logging.info("Starting training loop. Initial compile might take a while.")
    for step, batch in enumerate(tfds.as_numpy(ds_train)):
        state, train_loss = train_step(state, batch, rng_dropout)
        train_log.append(train_loss)

        if (step + 1) % steps_per_epoch == 0:
            for test_batch in tfds.as_numpy(ds_test):
                preds, test_loss = test_step(state, test_batch)
                test_log.append(test_loss)

            train_loss = np.mean(train_log)
            test_loss = np.mean(test_log)

            train_metrics.append(train_loss)
            test_metrics.append(test_loss)

            epoch = int((step + 1) // steps_per_epoch)

            logging.info(
                'Epoch {}: Train_loss = {:.6f}, Test_loss = {:.6f}'.format(
                    epoch, train_loss, test_loss))

            if epoch % 10 == 0:
                plot_prediction(config, preds[0, :, :, 0],
                                test_batch['decoder'][0,
                                :, :, 1], epoch, 0)
                plot_prediction(config, preds[0, :, :, 1],
                                test_batch['decoder'][0,
                                :, :, 1], epoch, 1)
                plot_prediction(config, preds[0, :, :, 2],
                                test_batch['decoder'][0,
                                :, :, 2], epoch, 2)

    # Data analysis plots
    plot_loss(config, train_metrics, test_metrics)

    # Save model
    ckpt = {'model': state}
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    save_args = orbax_utils.save_args_from_target(ckpt)
    orbax_checkpointer.save('nacaVIT', ckpt, save_args=save_args)
