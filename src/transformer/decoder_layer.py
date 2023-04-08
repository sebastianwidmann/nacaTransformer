#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Created By  : Sebastian Widmann
# Institution : TU Munich, Department of Aerospace and Geodesy
# Created Date: March 29, 2023
# version ='1.0'
# ---------------------------------------------------------------------------

from flax import linen as nn

from src.transformer.layers import MultiLayerPerceptron


class DecoderLayer(nn.Module):
    """
    Transformer decoder layer

    Attributes
    ----------
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
    num_heads: int
    dim_model: int
    dim_mlp: int
    dropout_rate: float = 0.1
    att_dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, input_decoder, input_encoder, deterministic):
        """

        Parameters
        ----------
        input_decoder:
        input_encoder:
        deterministic: bool
            If false, the attention weight is masked randomly using dropout,
            whereas if true, the attention weights are deterministic.

        Returns
        -------
        TODO: add dtype
            Output of encoder layer.
        """

        # Block 1: Norm, Masked Multi-Head Attention, Add
        x = nn.LayerNorm()(input_decoder)
        x = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            dropout_rate=self.att_dropout_rate,
        )(x, x, mask=nn.make_causal_mask(x), deterministic=deterministic)
        x = x + input_decoder

        # Block 2: Norm, Multi-Head Attention of Encoder and Decoder, Add
        y = nn.LayerNorm()(x)
        y = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            dropout_rate=self.att_dropout_rate,
        )(y, input_encoder, deterministic=deterministic)
        y = x + y

        # Block 3: Norm, Multilayer Perceptron, Add
        z = nn.LayerNorm()(y)
        z = MultiLayerPerceptron(
            dim_model=self.dim_model,
            dim_mlp=self.dim_mlp,
            dropout_rate=self.dropout_rate,
        )(z, deterministic=deterministic)

        return y + z
