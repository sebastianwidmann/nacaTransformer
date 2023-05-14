# ----------------------------------------------------------------------------
# Created By  : Sebastian Widmann
# Institution : TU Munich, Department of Aerospace and Geodesy
# Created Date: March 29, 2023
# version ='1.0'
# ---------------------------------------------------------------------------

import flax.linen as nn


class MultiLayerPerceptron(nn.Module):
    """
    Multilayer perceptron layer

    Attributes
    ----------
    hidden_size: int
        dimensionality of embeddings
    dim_mlp: int
        dimensionality of multilayer perceptron layer
    dropout_rate: float
        Dropout rate. Float between 0 and 1.
    """

    hidden_size: int
    dim_mlp: int
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, x, *, deterministic):
        """
        Applies MLP layer on the inputs.

        Parameters
        ----------
        x: jnp.ndarray
            Input of MLP layer
        deterministic: bool
            If false, the attention weight is masked randomly using dropout,
            whereas if true, the attention weights are deterministic.

        Returns
        -------
            Output of MLP layer.
        """

        x = nn.Dense(features=self.dim_mlp)(x)
        x = nn.gelu(x)
        x = nn.Dropout(rate=self.dropout_rate, deterministic=deterministic)(x)
        x = nn.Dense(features=self.hidden_size)(x)
        x = nn.Dropout(rate=self.dropout_rate, deterministic=deterministic)(x)
        return x
