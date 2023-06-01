# ----------------------------------------------------------------------------
# Created By  : Sebastian Widmann
# Institution : TU Munich, Department of Aerospace and Geodesy
# Created Date: March 29, 2023
# version ='1.0'
# ---------------------------------------------------------------------------

from flax import linen as nn
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
        y = self.decoder(**self.config, name='Decoder')(y, x, train=train)

        return y
