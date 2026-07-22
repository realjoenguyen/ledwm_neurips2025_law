# class VectorQuantizer(nj.Module):
#     def __init__(self, codes=512, embed=32):
#         self.codes = codes
#         self.book = nj.Variable(
#             lambda: jax.random.normal(nj.rng(), (self.codes, embed), jnp.float32)
#         )
# class VectorQuantizer(nj.Module):
#     def __init__(self, codes=512, embed=32):
#         self.codes = codes
#         self.book = nj.Variable(
#             lambda: jax.random.normal(nj.rng(), (self.codes, embed), jnp.float32)
#         )

#     def __call__(self, inputs):
#         book = self.book.read()
#         book /= jnp.linalg.norm(book, 2, -1, True)
#         flat = inputs.reshape((-1, inputs.shape[-1]))
#         flat /= jnp.linalg.norm(flat, 2, -1, True)
#         flat2 = (flat**2).sum(-1, keepdims=True)
#         book2 = (book**2).sum(-1, keepdims=True).T
#         dist = flat2 - 2 * (flat @ book.T) + book2
#         indices = jnp.argmin(dist, -1).reshape(inputs.shape[:-1])
#         outputs = book[indices]
#         outputs = inputs + sg(outputs - inputs)
#         return outputs, indices
#     def __call__(self, inputs):
#         book = self.book.read()
#         book /= jnp.linalg.norm(book, 2, -1, True)
#         flat = inputs.reshape((-1, inputs.shape[-1]))
#         flat /= jnp.linalg.norm(flat, 2, -1, True)
#         flat2 = (flat**2).sum(-1, keepdims=True)
#         book2 = (book**2).sum(-1, keepdims=True).T
#         dist = flat2 - 2 * (flat @ book.T) + book2
#         indices = jnp.argmin(dist, -1).reshape(inputs.shape[:-1])
#         outputs = book[indices]
#         outputs = inputs + sg(outputs - inputs)
#         return outputs, indices

#     def embed(self, indices):
#         book = self.book.read()
#         book /= jnp.linalg.norm(book, 2, -1, True)
#         return book[indices]
#     def embed(self, indices):
#         book = self.book.read()
#         book /= jnp.linalg.norm(book, 2, -1, True)
#         return book[indices]

#     def loss(self, inputs, indices, beta=0.25):
#         inputs = inputs.astype(jnp.float32)
#         embed = self.embed(indices).astype(jnp.float32)
#         loss_enc = ((sg(embed) - inputs) ** 2).mean(-1)
#         loss_book = ((embed - sg(inputs)) ** 2).mean(-1)
#         return loss_enc + beta * loss_book
#     def loss(self, inputs, indices, beta=0.25):
#         inputs = inputs.astype(jnp.float32)
#         embed = self.embed(indices).astype(jnp.float32)
#         loss_enc = ((sg(embed) - inputs) ** 2).mean(-1)
#         loss_book = ((embed - sg(inputs)) ** 2).mean(-1)
#         return loss_enc + beta * loss_book


from ledwm import ninjax as nj
from ledwm.nets.Attention import Attention
from ledwm.nets.Linear import LinearAct
from ledwm.nets.Norm import Norm
from ledwm.nets import get_act


import jax.numpy as jnp
import numpy as np


class Block(nj.Module):
    def __init__(
        self,
        size,
        groups=8,
        heads=8,
        act="gelu",
        norm="layer",
        winit="normal",
        fan="avg",
    ):
        assert norm == "layer", norm
        assert size % groups == 0, (size, groups)
        assert (size // groups) % heads == 0, (size, groups, heads)
        self.size = size
        self.act = get_act(act)
        self.groups = groups
        self.heads = heads
        self.kw = dict(winit=winit, fan=fan)

    def __call__(self, x):
        if x.shape[-1] % self.groups != 0:
            want = int(np.ceil(x.shape[-1] / self.groups) * self.groups)
            missing = want - x.shape[-1]
            x = jnp.concatenate([x, x[..., :missing]], -1)
            assert x.shape[-1] % self.groups == 0, (x.shape, self.groups)

        embed = self.size // self.groups
        x = x.reshape((*x.shape[:-1], self.groups, x.shape[-1] // self.groups))
        if x.shape[-1] != embed:
            x = self.get("proj", LinearAct, embed, **self.kw)(x)
        skip = x
        x = self.get("norm1", Norm, "layer")(x)
        dim = embed // self.heads
        x = self.get("attn1", Attention, self.heads, dim, **self.kw)(x, x, x)
        x += skip
        skip = x
        x = self.get("norm2", Norm, "layer")(x)
        x = self.get("linear1", LinearAct, embed, **self.kw)(x)
        x = self.act(x)
        x = self.get("linear2", LinearAct, embed, **self.kw)(x)
        x += skip
        x = x.reshape((*x.shape[:-2], self.size))
        return x
