import flax.linen as nn
from ledwm import jaxutils, ninjax as nj
from ledwm.nets.Dist import Dist
from ledwm.nets.Linear import LinearAct
from ledwm.nets.Input import Input


class MLP(nj.Module):
    def __init__(
        self,
        shape,  # if shape then output Dists
        layers,
        units,
        inputs=["tensor"],
        dims=None,
        symlog_inputs=False,
        dropout=0,
        dropout_inputs=None,
        **kw,  # store activation and norm here
    ):
        assert shape is None or isinstance(shape, (int, tuple, dict)), shape
        if isinstance(shape, int):
            shape = (shape,)
        self._shape = shape
        self._layers = layers
        self._units = units
        self._inputs = Input(inputs, dims=dims)
        self._symlog_inputs = symlog_inputs
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
        self._dense = {k: v for k, v in kw.items() if k not in distkeys}
        self._dist = {k: v for k, v in kw.items() if k in distkeys}

        self._dropout_inputs = dropout_inputs
        self._dropout = dropout

    def __call__(self, inputs, step=None, training=True):  # (bs * bl, d_input)
        feat = self._inputs.__call__(
            inputs,
            dropout_inputs=self._dropout_inputs,
            dropout=self._dropout,
            training=training,
            step=step,
        )
        if self._symlog_inputs:
            feat = jaxutils.symlog(feat)

        x = jaxutils.cast_to_compute(feat)  # bs, bl, all_dims (after concat)
        x = x.reshape([-1, x.shape[-1]])  # (bs*bl, d_input)

        for i in range(self._layers):
            x = self.get(f"h{i}", LinearAct, self._units, **self._dense)(x)

        x = x.reshape(feat.shape[:-1] + (x.shape[-1],))  # bs, bl, d_mlp

        if self._shape is None:
            return x

        elif isinstance(self._shape, tuple):  # for messenger tuple: (5,), reward
            return self._out("out", self._shape, x, step)

        elif isinstance(self._shape, dict):  # decoder: {"token_embed": (512,)}
            return {k: self._out(k, v, x, step) for k, v in self._shape.items()}

        else:
            raise ValueError(self._shape)

    def _out(self, name, shape, x, step=None):
        return self.get(f"dist_{name}", Dist, shape, **self._dist)(
            x, step
        )  # bs, bl, stoch, d_out
