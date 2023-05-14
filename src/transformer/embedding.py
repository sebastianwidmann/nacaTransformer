# ----------------------------------------------------------------------------
# Created By  : Sebastian Widmann
# Institution : TU Munich, Department of Aerospace and Geodesy
# Created Date: March 29, 2023
# version ='1.0'
# ---------------------------------------------------------------------------

from flax import linen as nn
import jax.numpy as jnp


class PatchEmbedding(nn.Module):
    """ Patch embedding.

    Attributes
    ----------
    patches: tuple
        number of patches (x_patches, y_patches)
    hidden_size: int
        number of features in PatchEmbedding
    """

    patches: int
    hidden_size: int

    @nn.compact
    def __call__(self, x):
        """ Applies patch embedding on the inputs.

        Parameters
        ----------
        x: jnp.ndarray
            Inputs of layer.

        Returns
        -------
        Flattened output of input. Shape = (batch_size, num_patches,
        num_channels)

        """

        x = nn.Conv(
            features=self.hidden_size,
            kernel_size=self.patches,
            strides=self.patches,
            padding='SAME',
        )(x)

        b, h, w, c = x.shape

        return x.reshape([b, h * w, c])


class PositionEmbedding(nn.Module):
    """ Position embedding. """

    @nn.compact
    def __call__(self, x):
        """ Applies position embedding on the inputs.

        Parameters
        ----------
        x: jnp.ndarraytwit
            Patch embedding layer.

        Returns
        -------
        Flattened output of input. Shape = (batch_size, num_patches,
        num_channels)

        """

        # x.shape is (batch_size, num_patches, emb_dim)
        y = self.param(
            'PositionEmbedding',
            nn.initializers.normal(),
            (1, x.shape[1], x.shape[2]),
        )

        return x + y
