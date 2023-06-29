# ----------------------------------------------------------------------------
# Created By  : Sebastian Widmann
# Institution : TU Munich, Department of Aerospace and Geodesy
# Created Date: March 29, 2023
# version ='1.0'
# ---------------------------------------------------------------------------

from flax import linen as nn

from src.transformer.embedding import PatchEmbedding, PositionEmbedding
from src.transformer.encoder_layer import EncoderLayer


class Encoder(nn.Module):
    """ Transformer encoder.

    Attributes
    ----------
    img_size: tuple
        number of pixels per dimension of image (x, y)
    patch_size: tuple
        number of pixels per patch dimension
    hidden_size: int
        dimensionality of embeddings
    num_layers: int
        number of layers
    num_heads: int
        number of heads in nn.MultiHeadDotProductAttention
    dim_mlp: int
        dimensionality of multilayer perceptron layer
    dropout_rate: float
        Dropout rate. Float between 0 and 1.
    att_dropout_rate: float
        Dropout rate of attention layer. Float between 0 and 1.
    """

    img_size: tuple
    patch_size: tuple
    hidden_size: int
    num_layers: int
    num_heads: int
    dim_mlp: int
    dropout_rate: float = 0.1
    att_dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, x, *, train):
        """ Applies transformer model on the inputs.

        Parameters
        ----------
        x: jnp.ndarray
            Inputs of encoder layer.
        train: bool
            Set to 'True' when training.

        Returns
        -------
        Output of transformer encoder.

        """

        x = PatchEmbedding(
            self.patch_size,
            self.hidden_size,
            name='PatchEmbedding'
        )(x)

        x = PositionEmbedding(name='PositionEmbedding')(x)

        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not train)

        for lyr in range(self.num_layers):
            x = EncoderLayer(
                num_heads=self.num_heads,
                hidden_size=self.hidden_size,
                dim_mlp=self.dim_mlp,
                dropout_rate=self.dropout_rate,
                att_dropout_rate=self.att_dropout_rate,
                name='Layer{}'.format(lyr),
            )(x, deterministic=not train)

        x = nn.LayerNorm()(x)

        return x
