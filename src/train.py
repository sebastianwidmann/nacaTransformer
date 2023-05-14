# ----------------------------------------------------------------------------
# Created By  : Sebastian Widmann
# Institution : TU Munich, Department of Aerospace and Geodesy
# Created Date: March 21, 2023
# version ='1.0'
# ---------------------------------------------------------------------------

from absl import logging
from flax import jax_utils
from flax.training import train_state
import jax
import jax.numpy as jnp
from ml_collections import ConfigDict
import optax
import tensorflow as tf
import tensorflow_datasets as tfds
from tqdm import tqdm
from typing import Any

from time import time, sleep

from src.transformer.input_pipeline import get_data_from_tfds
from src.transformer.vit import VisionTransformer

PRNGKey = Any


def init_train_state(config: ConfigDict, rng: PRNGKey) -> \
        train_state.TrainState:
    # Build VisionTransformer architecture
    model = VisionTransformer(config.vit)

    # Generate PRNGs
    rng_params, rng_dropout = jax.random.split(rng)
    init_rngs = {'params': rng_params,
                 'dropout': rng_dropout,
                 }

    # # Initialise model and use JIT to reside params in CPU memory
    init_x = (config.batch_size, *config.preprocess.resolution, 2)
    init_y = (config.batch_size, *config.preprocess.resolution, 3)

    variables = jax.jit(
        lambda: model.init(init_rngs,
                           jnp.ones(shape=init_x),
                           jnp.ones(shape=init_y),
                           train=False),
        backend='cpu',
    )()

    # Setup optimizer
    tx = optax.adam(learning_rate=config.learning_rate)
    # tx = optax.adamw(learning_rate=config.learning_rate,
    #                  weight_decay=config.weight_decay)

    # Initialise train state
    state = train_state.TrainState.create(
        apply_fn=model.apply,
        tx=tx,
        params=variables['params'],
    )

    return state


@jax.jit
def train_step(state: train_state.TrainState, x_batch: jnp.ndarray, y_batch:
jnp.ndarray, rng: PRNGKey) -> train_state.TrainState:
    key_dropout = jax.random.fold_in(rng, state.step)

    def loss_fn(params):
        preds = state.apply_fn({'params': params},
                               x_batch,
                               y_batch,
                               train=True,
                               rngs={'dropout': key_dropout},
                               )

        loss = optax.squared_error(preds, y_batch).mean()

        return loss, preds

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (_, preds), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)

    return state


def train_and_evaluate(config: ConfigDict):
    logging.info("Initialising airfoilMNIST dataset.")
    ds_train = get_data_from_tfds(config=config, mode='train')
    ds_test = get_data_from_tfds(config=config, mode='test')

    # model = VisionTransformer(config.vit)
    # print(model.tabulate(jax.random.PRNGKey(0), jnp.ones((4, 200, 200, 3)),
    # jnp.ones((4, 200, 200, 3)), train=True))

    rng = jax.random.PRNGKey(0)

    state = init_train_state(config, rng)

    for _ in range(config.num_epochs):
        for batch in tqdm(tfds.as_numpy(ds_train), desc='Batch', position=1,
                          leave=False):
            x, y = batch['encoder'], batch['decoder']

            state = train_step(state, x, y, rng)
