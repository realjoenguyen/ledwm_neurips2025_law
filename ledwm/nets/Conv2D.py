from ledwm import jaxutils, ninjax as nj
from ledwm.nets.Initializer import Initializer
from ledwm.nets.Norm import Norm
from ledwm.nets import get_act


import jax
import jax.numpy as jnp
import numpy as np

SCALE_INIT = 2
BIAS_INIT = 5


class Conv2D(nj.Module):
    def __init__(
        self,
        depth,
        kernel,
        stride=1,
        transp=False,
        act="none",
        norm="none",
        pad="same",
        bias=True,
        # bias_for_last=False,  # only for the last layer
        preact=False,
        winit="uniform",
        fan="avg",
        # shortcut=False,
    ):
        self._depth = depth
        self._kernel = kernel
        self._stride = stride
        self._transp = transp
        self._act = get_act(act)
        self._norm = Norm(norm, name="norm")
        self._pad = pad.upper()
        self._bias = bias and (preact or norm == "none")
        # self._bias_for_last = bias_for_last
        self._preact = preact
        self._winit = winit
        self._fan = fan
        # self._shortcut = shortcut
        # print(f"{shortcut=}")

    def __call__(self, hidden, style=None):
        if self._preact:
            hidden = self._norm(hidden, style)
            hidden = self._act(hidden)
            hidden = self._layer(hidden)
        else:
            hidden = self._layer(hidden)
            hidden = self._norm(hidden, style)
            hidden = self._act(hidden)
        return hidden

    def _layer(self, x):
        if self._transp:
            shape = (self._kernel, self._kernel, self._depth, x.shape[-1])
            kernel = self.get(
                "kernel",
                Initializer(self._winit, scale=SCALE_INIT, fan=self._fan),
                shape,
            )
            kernel = jaxutils.cast_to_compute(kernel)
            x = jax.lax.conv_transpose(
                x,
                kernel,
                (self._stride, self._stride),
                self._pad,
                dimension_numbers=("NHWC", "HWOI", "NHWC"),
            )

        else:
            shape = (self._kernel, self._kernel, x.shape[-1], self._depth)
            kernel = self.get(
                "kernel",
                Initializer(self._winit, scale=SCALE_INIT, fan=self._fan),
                shape,
            )
            kernel = jaxutils.cast_to_compute(kernel)
            x = jax.lax.conv_general_dilated(
                x,
                kernel,
                (self._stride, self._stride),
                self._pad,
                dimension_numbers=("NHWC", "HWIO", "NHWC"),
            )

            # if self._shortcut:
            #     x += x_shortcut

        if self._bias:
            # if self._bias_for_last:
            #     print("use bias -1 for the last layer")
            #     bias = self.get(
            #         "bias",
            #         lambda *args: jnp.ones(self._depth, np.float32) * (-BIAS_OFFSET),
            #     )
            # else:
            bias = self.get("bias", jnp.zeros, self._depth, np.float32)
            bias = jaxutils.cast_to_compute(bias)
            x += bias
        return x
