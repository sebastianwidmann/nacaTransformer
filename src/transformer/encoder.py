#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Created By  : Sebastian Widmann
# Institution : TU Munich, Department of Aerospace and Geodesy
# Created Date: March 29, 2023
# version ='1.0'
# ---------------------------------------------------------------------------

from flax import linen as nn

from src.transformer.encoder_layer import EncoderLayer


class Encoder(nn.Module):
    """ Transformer encoder.

    Attributes
    ----------
    num_layers: int
        number of layers
    pos_embedding: bool = True
        Add learned positional embeddings to the inputs
    num_heads: int
        number of heads in nn.MultiHeadDotProductAttention
    dim_model: int
        dimensionality of embeddings
    dim_mlp: int
        dimensionality of multilayer perceptron layer
    dropout_rate: float
        Dropout rate. Float between 0 and 1.
    att_dropout_rate: float
        Dropout rate of attention layer. Float between 0 and 1.
    """

    num_layers: int
    pos_embedding: bool = True
    num_heads: int
    dim_model: int
    dim_mlp: int
    dropout_rate: float = 0.1
    att_dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, x, train):
        """ Applies transformer model on the inputs.

        Parameters
        ----------
        x: TODO: add dtype
            Inputs of encoder layer.
        train: bool
            Set to 'True' when training.

        Returns
        -------
        Output of transformer encoder.

        """

        # TODO: add positional embedding call

        for lyr in range(self.num_layers):
            x = EncoderLayer(
                num_heads=self.num_heads,
                dim_model=self.dim_model,
                dim_mlp=self.dim_mlp,
                dropout_rate=self.dropout_rate,
                att_dropout_rate=self.att_dropout_rate,
            )(x, deterministic=not train)

        encoder = nn.LayerNorm(name='encoder_norm')(x)

        return encoder
