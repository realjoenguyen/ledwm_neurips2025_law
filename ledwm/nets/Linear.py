from ledwm import jaxutils, ninjax as nj
from ledwm.nets.Conv2D import BIAS_INIT
from ledwm.nets.Initializer import Initializer
from ledwm.nets.Norm import Norm
from ledwm.nets import get_act


import jax.numpy as jnp
import numpy as np


# norm = None, act = None
class LinearAct(nj.Module):
    def __init__(
        self,
        units,
        act="none",
        norm="none",
        bias=True,
        outscale=1.0,
        outnorm=False,
        winit="normal",
        fan="avg",
        bias_last=False,
        batch_dims=None,
    ):
        self._units = tuple(units) if hasattr(units, "__len__") else (units,)
        self._act = get_act(act)
        self._norm = norm
        self._bias = bias and norm == "none"
        self._outscale = outscale
        self._outnorm = outnorm
        self._winit = winit
        self._fan = fan
        self._bias_last = bias_last
        self._batch_dims = batch_dims

    def __call__(self, x):
        # product of all self._units
        shape = (x.shape[-1], np.prod(self._units))
        kernel = self.get(
            "kernel", Initializer(self._winit, self._outscale, fan=self._fan), shape
        )
        kernel = jaxutils.cast_to_compute(kernel)
        x = x @ kernel
        if self._bias:
            if self._bias_last:
                print(f"linear.config | bias_last=true | bias_init={BIAS_INIT}")
                bias = self.get(
                    "bias",
                    lambda *args: jnp.ones(np.prod(self._units), np.float32)
                    * (-BIAS_INIT),
                )
            else:
                bias = self.get("bias", jnp.zeros, np.prod(self._units), np.float32)
            bias = jaxutils.cast_to_compute(bias)
            x += bias

        if len(self._units) > 1:
            x = x.reshape(x.shape[:-1] + self._units)

        x = self.get("norm", Norm, self._norm, self._batch_dims)(x)
        x = self._act(x)
        return x


# %%


def test_reshape():
    import jax.numpy as jnp
    import numpy as np

    # import deepcopy
    # set gpu id
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    # random jax array: 200, 1600 , not ones or zeros
    # x = jnp.array(np.random.rand(200, 1600))
    x = np.random.rand(200, 1600)
    print(x.shape)
    # x_raw deep copy from x
    # x_raw = jnp.copy(x)
    x_raw = np.copy(x)
    x = x.reshape(200, 5, 5, 64).reshape(200, 1600)
    print(x.shape)
    # check if x is the same as x_raw
    # assert jnp.all(x == x_raw)
    # assert np.all(x == x_raw)


# test_reshape()
