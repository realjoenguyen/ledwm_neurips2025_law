from ledwm import jaxutils, ninjax as nj
from ledwm.nets.Conv2D import Conv2D
from ledwm.nets.Linear import LinearAct


import jax
import jax.numpy as jnp
import numpy as np


class ImageDecoderStyle(nj.Module):
    def __init__(self, shape, depth, blocks, resize, minres, sigmoid, **kw):
        self._shape = shape
        self._depth = depth
        self._blocks = blocks
        self._resize = resize
        self._minres = minres
        self._sigmoid = sigmoid
        self._kw = kw

    def __call__(self, x):
        stages = int(np.log2(self._shape[-2]) - np.log2(self._minres))

        style = x
        for i in range(4):
            style = self.get(f"style{i}", LinearAct, 1024, **self._kw)(style)
        style = x
        for i in range(4):
            style = self.get(f"style{i}", LinearAct, 1024, **self._kw)(style)

        depth = self._depth * 2 ** (stages - 1)
        x = jaxutils.cast_to_compute(x)
        x = self.get("in", LinearAct, (self._minres, self._minres, depth))(x)
        for i in range(stages):
            for j in range(self._blocks):
                skip = x
                kw = {**self._kw, "preact": True}
                s1 = self.get(f"s{i}b{j}s1", LinearAct, 2 * depth)(style)
                s2 = self.get(f"s{i}b{j}s2", LinearAct, 2 * depth)(style)
                s1 = jnp.split(s1[..., None, None, :], 2, -1)
                s2 = jnp.split(s2[..., None, None, :], 2, -1)
                x = self.get(f"s{i}b{j}c1", Conv2D, depth, 3, **kw)(x, s1)
                x = self.get(f"s{i}b{j}c2", Conv2D, depth, 3, **kw)(x, s2)
                x += skip
                # print(x.shape)
            depth //= 2
            kw = {**self._kw, "preact": False}
            if i == stages - 1:
                kw = {}
                depth = self._shape[-1]
            if self._resize == "stride":
                s = None
                if self._blocks == 0:
                    s = self.get(f"s{i}s", LinearAct, 2 * depth)(style)
                    s = jnp.split(s[..., None, None, :], 2, -1)
                x = self.get(f"s{i}res", Conv2D, depth, 4, 2, transp=True, **kw)(x, s)
            elif self._resize == "stride3":
                s = 3 if i == stages - 1 else 2
                k = 5 if i == stages - 1 else 4
                x = self.get(f"s{i}res", Conv2D, depth, k, s, transp=True, **kw)(x)
            elif self._resize == "resize":
                x = jnp.repeat(jnp.repeat(x, 2, 1), 2, 2)
                x = self.get(f"s{i}res", Conv2D, depth, 3, 1, **kw)(x)
            else:
                raise NotImplementedError(self._resize)
        if max(x.shape[1:-1]) > max(self._shape[:-1]):
            padh = (x.shape[1] - self._shape[0]) / 2
            padw = (x.shape[2] - self._shape[1]) / 2
            x = x[:, int(np.ceil(padh)) : -int(padh), :]
            x = x[:, :, int(np.ceil(padw)) : -int(padw)]
        # print(x.shape)
        assert x.shape[-3:] == self._shape, (x.shape, self._shape)
        if self._sigmoid:
            x = jax.nn.sigmoid(x)
        else:
            x = x + 0.5
        return x
        depth = self._depth * 2 ** (stages - 1)
        x = jaxutils.cast_to_compute(x)
        x = self.get("in", LinearAct, (self._minres, self._minres, depth))(x)
        for i in range(stages):
            for j in range(self._blocks):
                skip = x
                kw = {**self._kw, "preact": True}
                s1 = self.get(f"s{i}b{j}s1", LinearAct, 2 * depth)(style)
                s2 = self.get(f"s{i}b{j}s2", LinearAct, 2 * depth)(style)
                s1 = jnp.split(s1[..., None, None, :], 2, -1)
                s2 = jnp.split(s2[..., None, None, :], 2, -1)
                x = self.get(f"s{i}b{j}c1", Conv2D, depth, 3, **kw)(x, s1)
                x = self.get(f"s{i}b{j}c2", Conv2D, depth, 3, **kw)(x, s2)
                x += skip
                # print(x.shape)
            depth //= 2
            kw = {**self._kw, "preact": False}
            if i == stages - 1:
                kw = {}
                depth = self._shape[-1]
            if self._resize == "stride":
                s = None
                if self._blocks == 0:
                    s = self.get(f"s{i}s", LinearAct, 2 * depth)(style)
                    s = jnp.split(s[..., None, None, :], 2, -1)
                x = self.get(f"s{i}res", Conv2D, depth, 4, 2, transp=True, **kw)(x, s)
            elif self._resize == "stride3":
                s = 3 if i == stages - 1 else 2
                k = 5 if i == stages - 1 else 4
                x = self.get(f"s{i}res", Conv2D, depth, k, s, transp=True, **kw)(x)
            elif self._resize == "resize":
                x = jnp.repeat(jnp.repeat(x, 2, 1), 2, 2)
                x = self.get(f"s{i}res", Conv2D, depth, 3, 1, **kw)(x)
            else:
                raise NotImplementedError(self._resize)
        if max(x.shape[1:-1]) > max(self._shape[:-1]):
            padh = (x.shape[1] - self._shape[0]) / 2
            padw = (x.shape[2] - self._shape[1]) / 2
            x = x[:, int(np.ceil(padh)) : -int(padh), :]
            x = x[:, :, int(np.ceil(padw)) : -int(padw)]
        # print(x.shape)
        assert x.shape[-3:] == self._shape, (x.shape, self._shape)
        if self._sigmoid:
            x = jax.nn.sigmoid(x)
        else:
            x = x + 0.5
        return x
