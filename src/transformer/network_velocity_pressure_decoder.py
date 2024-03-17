# ----------------------------------------------------------------------------
# Created By  : Sebastian Widmann
# Institution : TU Munich, Department of Aerospace and Geodesy
# Created Date: March 29, 2023
# version ='1.0'
# ---------------------------------------------------------------------------

from flax import linen as nn
import jax.numpy as jnp
from typing import Type
from ml_collections import ConfigDict

from src.transformer.encoder import Encoder
from src.transformer.decoder import Decoder


class VisionTransformer(nn.Module):
    config: Type[ConfigDict]
    encoder: Type[nn.Module] = Encoder
    decoder: Type[nn.Module] = Decoder

    @nn.compact
    def __call__(self, x, y, *, train):
        x = self.encoder(**self.config, name='Encoder')(x, train=train)

        y_u = y[:,:,:,0:2]
        y_p1 = y[:,:,:,2]

        y_p = jnp.expand_dims(y_p1, axis=-1)

        y_u = self.decoder(**self.config, name='Decoder_u')(y_u, x, train=train)
        y_p = self.decoder(**self.config, name='Decoder_p')(y_p, x, train=train)

        y = jnp.concatenate(arrays=(y_u,y_p), axis=-1) 

        return y
