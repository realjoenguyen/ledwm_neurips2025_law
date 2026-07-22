from ledwm import ninjax as nj
from ledwm.nets import f32


import jax
import jax.numpy as jnp
import numpy as np


class Initializer:
    def __init__(self, dist="normal", scale=1.0, fan="avg"):
        self.scale = scale
        self.dist = dist
        self.fan = fan

    def __call__(self, shape):
        if self.scale == 0.0:
            value = jnp.zeros(shape, f32)

        elif self.dist == "uniform":
            fanin, fanout = self._fans(shape)
            denoms = {"avg": (fanin + fanout) / 2, "in": fanin, "out": fanout}
            scale = self.scale / denoms[self.fan]
            limit = np.sqrt(3 * scale)
            value = jax.random.uniform(nj.rng(), shape, f32, -limit, limit)

        elif self.dist == "normal":
            fanin, fanout = self._fans(shape)
            denoms = {"avg": np.mean((fanin, fanout)), "in": fanin, "out": fanout}
            scale = self.scale / denoms[self.fan]
            std = np.sqrt(scale) / 0.87962566103423978
            value = std * jax.random.truncated_normal(nj.rng(), -2, 2, shape, f32)

        elif self.dist == "ortho":
            nrows, ncols = shape[-1], np.prod(shape) // shape[-1]
            matshape = (nrows, ncols) if nrows > ncols else (ncols, nrows)
            mat = jax.random.normal(nj.rng(), matshape, f32)
            qmat, rmat = jnp.linalg.qr(mat)
            qmat *= jnp.sign(jnp.diag(rmat))
            qmat = qmat.T if nrows < ncols else qmat
            qmat = qmat.reshape(nrows, *shape[:-1])
            value = self.scale * jnp.moveaxis(qmat, 0, -1)

        else:
            raise NotImplementedError(self.dist)
        return value

    def _fans(self, shape):
        if len(shape) == 0:
            return 1, 1
        elif len(shape) == 1:
            return shape[0], shape[0]
        elif len(shape) == 2:
            return shape
        else:
            space = int(np.prod(shape[:-2]))
            return shape[-2] * space, shape[-1] * space
