import jax
from ledwm.nets.Input import Input
from ledwm.nets.Linear import LinearAct
from ledwm.nets.Norm import Norm
from ledwm import jaxutils, ninjax as nj
from typing import Any

from ledwm.nets.Dist import Dist


class ResidualMLP(nj.Module):
    """
    linear -> [[LayerNorm -> Linear -> ReLu -> Linear] + (Skip)]x num_blocks ->Layernorm
    """

    def __init__(
        self,
        units,
        inputs=["tensor"],
        num_blocks=1,
        dims=None,
        batch_dims=None,
        norm="rsm",
        symlog_inputs=False,
    ):
        # turn all to private variables
        self._num_blocks = num_blocks
        self._norm = norm
        self._batch_dims = batch_dims
        self._units = units
        self._inputs = Input(inputs, dims=dims)
        self._symlog_inputs = symlog_inputs

    def __call__(self, inputs):
        feat = self._inputs.__call__(inputs)
        if self._symlog_inputs:
            feat = jaxutils.symlog(feat)
        x = jaxutils.cast_to_compute(feat)
        x = x.reshape([-1, x.shape[-1]])  # (bs*bl, d_input)

        print(f"residual_mlp.start | blocks={self._num_blocks}")
        x = self.get("linear_inp", LinearAct, self._units)(x)
        print(f"residual_mlp.tensor | stage=input_linear | shape={x.shape}")
        for t in range(self._num_blocks):
            x = self.get(
                f"residual_block_{t}",
                ResidualBlock,
                self._units,
                self._norm,
                self._batch_dims,
            )(x, t)
            print(f"residual_mlp.tensor | stage=block | index={t} | shape={x.shape}")
        x = x.reshape(feat.shape[:-1] + (x.shape[-1],))  # bs, bl, d_input
        return x


class ResidualDist(ResidualMLP):
    def __init__(
        self,
        shape,
        units,
        inputs=["tensor"],
        num_blocks=1,
        dims=None,
        batch_dims=None,
        norm="rsm",
        symlog_inputs=False,
        **kw,
    ):
        # init super
        super().__init__(
            units,
            inputs,
            num_blocks,
            dims,
            batch_dims,
            norm,
            symlog_inputs,
        )

        assert shape is None or isinstance(shape, (int, tuple, dict)), shape
        if isinstance(shape, int):
            shape = (shape,)
        self._shape = shape
        distkeys = (
            "dist",
            "outscale",
            "minstd",
            "maxstd",
            "outnorm",
            "unimix",
            "bins",
            "bound",
            "unimix_decay",
            "discreet_values",
        )
        self._dist = {k: v for k, v in kw.items() if k in distkeys}

    def __call__(self, inputs, step=None, training=True):
        x = super().__call__(inputs)

        if self._shape is None:
            return x

        elif isinstance(self._shape, tuple):  # for messenger tuple: (5,), reward
            return self._out("out", self._shape, x, step)

        elif isinstance(self._shape, dict):  # decoder: {"token_embed": (512,)}
            return {k: self._out(k, v, x, step) for k, v in self._shape.items()}

        else:
            raise ValueError(self._shape)

    def _out(self, name, shape, x, step=None):
        # bs, bl, stoch, d_out
        return self.get(f"dist_{name}", Dist, shape, **self._dist)(x, step)


class ResidualBlock(nj.Module):
    """
    residual block: [LayerNorm -> Linear -> ReLu -> Linear] + (Skip)
    """

    def __init__(self, units, norm="layer", batch_dims=None):
        self._units = units
        self._norm = norm
        self._batch_dims = batch_dims

    def __call__(self, x, t: int):
        x_input = x
        x = self.get(f"norm1_{t}", Norm, self._norm, self._batch_dims)(x)
        x = self.get(f"linear1_{t}", LinearAct, self._units)(x)
        x = jax.nn.relu(x)
        x = self.get(f"linear2_{t}", LinearAct, self._units)(x)
        x = self.get(f"norm2_{t}", Norm, self._norm, self._batch_dims)(x)
        return x + x_input
