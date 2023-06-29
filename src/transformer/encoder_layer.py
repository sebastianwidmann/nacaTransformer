# ----------------------------------------------------------------------------
# Created By  : Sebastian Widmann
# Institution : TU Munich, Department of Aerospace and Geodesy
# Created Date: March 29, 2023
# version ='1.0'
# ---------------------------------------------------------------------------

from flax import linen as nn

from src.transformer.layers import MultiLayerPerceptron


class EncoderLayer(nn.Module):
    """ Transformer encoder layer.

    Attributes
    ----------
    num_heads: int
        number of heads in nn.MultiHeadDotProductAttention
    hidden_size: int
        dimensionality of embeddings
    dim_mlp: int
        dimensionality of multilayer perceptron layer
    dropout_rate: float
        Dropout rate. Float between 0 and 1.
    att_dropout_rate: float
        Dropout rate of attention layer. Float between 0 and 1.
    """

    num_heads: int
    hidden_size: int
    dim_mlp: int
    dropout_rate: float = 0.0
    att_dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, input_encoder, *, deterministic):
        """

        Parameters
        ----------
        input_encoder:
        deterministic: bool
            If false, the attention weight is masked randomly using dropout,
            whereas if true, the attention weights are deterministic.

        Returns
        -------
            Output of encoder layer.
        """

        # Block 1: Norm, Multi-Head Self-Attention, Add
        x = nn.LayerNorm()(input_encoder)

        x = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            dropout_rate=self.att_dropout_rate,
        )(x, x, deterministic=deterministic)

        x = x + input_encoder

        # Block 2: Norm, Multilayer Perceptron, Add
        y = nn.LayerNorm()(x)

        y = MultiLayerPerceptron(
            hidden_size=self.hidden_size,
            dim_mlp=self.dim_mlp,
            dropout_rate=self.dropout_rate,
        )(y, deterministic=deterministic)

        return x + y
