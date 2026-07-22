from ledwm import ninjax as nj
from ledwm.nets.Linear import LinearAct


import jax
import jax.numpy as jnp
import numpy as np

from ledwm.embodied.envs.MessengerSent import HIST_LEN

# HIST = 3


# class AttentionAND(nj.Module):
#     def __init__(
#         self,
#         query_dim,
#         key_dim,
#         value_dim,
#         winit="normal",
#         fan="avg",
#         entity_track_atten=False,
#         mlp_query_key=False,
#         heads=1,  # TODO del
#     ):
#         self.entity_track_atten = entity_track_atten
#         heads = 1
#         self.query_dim = query_dim // heads
#         self.key_dim = key_dim // heads
#         assert self.query_dim == self.key_dim
#         self.value_dim = value_dim // heads
#         self.kw = dict(winit=winit, fan=fan)
#         self.mlp_query_key = mlp_query_key
#         self.heads = 1
#         print("Attention: Use mlp for query and key")

#     def __call__(
#         self,
#         query1,
#         query2,
#         key,
#         value,
#         time_step,  # bs,
#         mask=None,
#     ):
#         qk_shape = (self.heads, self.query_dim)
#         query1 = self.get("query1", LinearAct, qk_shape, **self.kw)(query1)
#         query2 = self.get("query2", LinearAct, qk_shape, **self.kw)(query2)
#         key1 = self.get("key1", LinearAct, qk_shape, **self.kw)(key)
#         key2 = self.get("key2", LinearAct, qk_shape, **self.kw)(key)
#         value_shape = (self.heads, self.value_dim)
#         value = self.get("value", LinearAct, value_shape, **self.kw)(
#             value
#         )  # bs, Ne, heads, size

#         logits1 = jnp.einsum(
#             "...thd,...Thd->...htT",
#             query1,  # bs, Ne, heads, size
#             key1,  # bs, Ne, heads, size
#         )
#         # logits1 /= np.sqrt(self.query_dim).astype(key.dtype)  # bs, 1, Ne, Ne

#         logits2 = jnp.einsum(
#             "...thd,...Thd->...htT",
#             query2,  # bs, Ne, heads, size
#             key2,  # bs, Ne, heads, size
#         )
#         # logits2 /= np.sqrt(self.query_dim).astype(key.dtype)  # bs, 1, Ne, Ne

#         # logits_and = jnp.minimum(logits1, logits2)

#         if mask is not None:
#             assert mask.ndim == logits1.ndim
#             logits1 = jnp.where(mask, logits1, -np.inf)
#             logits2 = jnp.where(mask, logits2, -np.inf)
#             # logits_and = jnp.where(mask, logits_and, -np.inf)

#         # weights1 = jax.nn.softmax(logits1)  # bs, heads, Ne, Ne
#         # weights2 = jax.nn.softmax(logits2)  # bs, heads, Ne, Ne
#         # weights_and = jax.nn.softmax(logits_and)
#         # and operation
#         # weights = jnp.minimum(weights1, weights2)
#         # weights = jnp.where(
#         #     time_step[:, None, None, None] >= HIST_LEN - 1,
#         #     weights_and,
#         #     weights1,
#         # )
#         # logits = jnp.where(time_step[:, None, None, None] >= HIST, logits_and, logits1)
#         weights1 = jax.nn.sigmoid(logits1)
#         weights2 = jax.nn.sigmoid(logits2)
#         weights_and = jnp.minimum(weights1, weights2)
#         weights = jnp.where(
#             time_step[:, None, None, None] >= HIST_LEN - 1,
#             weights_and,
#             weights1,
#         )

#         x = jnp.einsum("...htT,...Thd->...thd", weights, value)
#         x = x.reshape((*x.shape[:-2], -1))  # bs, Ne, heads*size
#         x = self.get(
#             "out",
#             LinearAct,
#             (
#                 self.heads * self.size
#                 if hasattr(self, "size")
#                 else self.heads * self.value_dim
#             ),
#         )(x)

#         return {"x": x, "weights": weights, "weights1": weights1, "weights2": weights2}
