# ----------------------------------------------------------------------------
# Created By  : Sebastian Widmann
# Institution : TU Munich, Department of Aerospace and Geodesy
# Created Date: March 21, 2023
# version ='1.0'
# ---------------------------------------------------------------------------

from absl import logging
from flax import jax_utils
from flax.training import train_state, checkpoints
import jax
import jax.numpy as jnp
from ml_collections import ConfigDict
import numpy as np
import optax
from tqdm import tqdm
from typing import Any, Tuple

from time import time

from src.transformer.input_pipeline import get_data_from_tfds
from src.transformer.vit import VisionTransformer
from src.utilities.visualisation import plot_prediction

PRNGKey = Any


def init_train_state(config: ConfigDict,
                     rng: PRNGKey) -> train_state.TrainState:
    # Build VisionTransformer architecture
    model = VisionTransformer(config.vit)

    # Generate PRNGs
    rng_params, rng_dropout = jax.random.split(rng)

    init_rng = {'params': rng_params, 'dropout': rng_dropout}

    # Initialise model and use JIT to reside params in CPU memory
    variables = jax.jit(lambda: model.init(init_rng, jnp.ones(
        [config.batch_size, *config.vit.img_size, 1]), jnp.ones(
        [config.batch_size, *config.vit.img_size, 3]), train=False),
                        backend='cpu')()

    # Initialise train state
    tx = optax.adamw(learning_rate=config.learning_rate,
                     weight_decay=config.weight_decay)
    state = train_state.TrainState.create(
        apply_fn=model.apply, params=variables['params'], tx=tx)

    return state


# def loss_fn(state, batch):
#     preds = state.apply_fn(state.params, )

@jax.jit
def train_step(state: train_state.TrainState, batch: jnp.ndarray,
               rng: PRNGKey) -> Tuple[train_state.TrainState, Any]:
    key_dropout = jax.random.fold_in(rng, state.step)

    def loss_fn(params):
        preds = state.apply_fn({'params': params},
                               batch['encoder'], batch['decoder'],
                               train=True,
                               rngs={'dropout': key_dropout},
                               )

        loss = optax.squared_error(preds, batch['decoder']).mean()

        return loss, preds

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, preds), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)

    return state, loss


@jax.jit
def test_step(state: train_state.TrainState, batch: jnp.ndarray,
              rng: PRNGKey):
    key_dropout = jax.random.fold_in(rng, state.step)
    preds = state.apply_fn({'params': state.params},
                           batch['encoder'], batch['decoder'],
                           train=False,
                           rngs={'dropout': key_dropout},
                           )

    loss = optax.squared_error(preds, batch['decoder']).mean()

    return preds, loss


def train_and_evaluate(config: ConfigDict):
    logging.info("Initialising airfoilMNIST dataset.")
    ds_train = get_data_from_tfds(config=config, mode='train')
    ds_test = get_data_from_tfds(config=config, mode='test')

    rng = jax.random.PRNGKey(0)
    state = init_train_state(config, rng)

    for epoch in range(config.num_epochs):
        epoch_start = time()
        train_metrics = []
        for batch in ds_train:
            state, train_loss = train_step(state, batch, rng)
            train_metrics.append(train_loss)

        test_metrics = []
        for test_batch in ds_test:
            preds, test_loss = test_step(state, test_batch, rng)
            test_metrics.append(test_loss)

        epoch_end = time() - epoch_start

        train_loss = np.mean(train_metrics)
        test_loss = np.mean(test_metrics)

        print('Epoch {} in {:.2f} sec: Train_loss = {}, Test_loss = {}'.format(
            epoch, epoch_end, train_loss, test_loss))

    plot_prediction(config, preds[0, :, :, 1], test_batch['decoder'][0,
                                               :, :, 1], 0)
    plot_prediction(config, preds[0, :, :, 1], test_batch['decoder'][0,
                                               :, :, 1], 1)
    plot_prediction(config, preds[0, :, :, 2], test_batch['decoder'][0,
                                               :, :, 2], 2)

    # Save model
    checkpoints.save_checkpoint('nacaVIT', target=state, step=int(state.step))
