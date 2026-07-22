import jax
import jax.numpy as jnp
import numpy as np

from ledwm import ninjax as nj
import flax.linen as nn
from ledwm.jaxutils import apply_dropout_on


class Input:
    def __init__(self, keys=["tensor"], dims=None):
        """
        concat along the last dimension, based on len(dims)
        For example, if dims = 2, and a tensor has a shape of (3, 4, 5), the code will reshape it to (3, 20). The goal is to maintain consistency in dimensionality across different inputs.
        Args:
            keys (list or tuple, optional): List of keys. Defaults to ["tensor"].
            dims (None or any, optional): Dimensions. Defaults to None.
        """
        assert isinstance(keys, (list, tuple)), keys
        self._keys = tuple(keys)
        self._dims = dims or self._keys[0]

    def __call__(
        self,
        inputs,
        dropout_inputs=None,
        dropout=0,
        training=True,
        step=None,
    ):
        if not isinstance(inputs, dict):
            inputs = {"tensor": inputs}

        # inputs = raw_inputs.copy()
        assert isinstance(inputs, dict), inputs
        if "deter_layers" in inputs and "deter" in self._keys:
            # the last T in deter_layers: (*bs_dim, T, dim) -> (*bs_dim,  dim)
            inputs["deter"] = inputs["deter_layers"][..., -1, :]

        inputs.update({k: v for k, v in inputs.items() if k in self._keys})

        for key in self._keys:
            if key.startswith("softmax_"):
                inputs[key] = jax.nn.softmax(inputs[key[len("softmax_") :]])

        if not all(k in inputs for k in self._keys):
            needs = f"{{{', '.join(self._keys)}}}"
            found = f"{{{', '.join(inputs.keys())}}}"
            raise KeyError(f"Cannot find keys {needs} among inputs {found}.")

        if dropout > 0:
            dropout_fn = nj.FlaxModule(nn.Dropout, rate=dropout, name="dropout")
        else:
            dropout_fn = None

        values = []
        for k in self._keys:
            if dropout > 0 and dropout_inputs is not None and k in dropout_inputs:
                assert dropout > 0, dropout
                values.append(apply_dropout_on(inputs[k], dropout_fn, training, step))
            else:
                values.append(inputs[k])

        dims = len(inputs[self._dims].shape) if self._dims in inputs else self._dims

        for i, value in enumerate(values):
            if len(value.shape) > dims:
                values[i] = value.reshape(
                    value.shape[: dims - 1] + (np.prod(value.shape[dims - 1 :]),)
                )
        values = [x.astype(inputs[self._dims].dtype) for x in values]
        return jnp.concatenate(values, -1)
