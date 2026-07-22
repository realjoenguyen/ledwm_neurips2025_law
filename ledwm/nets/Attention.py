from ledwm import ninjax as nj
from ledwm.nets.Linear import LinearAct
import jax
import jax.numpy as jnp
import numpy as np
from ledwm.nets.MLP import MLP
import flax.linen as nn
from einops import rearrange, repeat
from jax.numpy import einsum

INF = 1000


class Attention(nj.Module):
    def __init__(
        self,
        query_dim,
        key_dim,
        value_dim,
        heads=1,
        size=None,
        winit="normal",
        fan="avg",
        entity_track_atten=False,
        mlp_query_key=False,
        norm="layer",
        temperature=1,
        dropout=0,
        gt_grounding=False,
        pos_head_seperate=False,
    ):
        self.heads = heads
        self.entity_track_atten = entity_track_atten
        self.query_dim = query_dim
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.kw = dict(winit=winit, fan=fan, norm=norm)
        self.mlp_query_key = mlp_query_key
        self.temperature = temperature
        if dropout > 0:
            self.dropout = nj.FlaxModule(
                nn.Dropout,
                rate=dropout,
                name="drop",
            )
        print(
            f"attention.config | "
            f"mlp_query_key={str(self.mlp_query_key).lower()}"
        )
        self.gt_grounding = gt_grounding
        self.pos_head_seperate = pos_head_seperate

    def __call__(
        self,
        query,  # bs, Ne, d
        key,
        value_input,
        mask=None,
        training=True,
        step=None,
        gt_grounding_scores=None,  # bs, Ne
        pos_dim=None,
    ):
        """
        return: (bs, Ne, heads*size), (bs, head, Ne, Ns)
        """
        if self.gt_grounding:
            assert gt_grounding_scores is not None
            atten_scores = repeat(gt_grounding_scores, "... n -> ... h n", h=self.heads)
            # turn into one-hot vector
            # bs, head, Ne, Ns
            atten_scores = jax.nn.one_hot(atten_scores, atten_scores.shape[-1])
            value_shape = (self.heads, self.value_dim)
            value = self.get("value", LinearAct, value_shape)(value_input)

        else:
            assert gt_grounding_scores is None
            QUERY_KEY_HEADS = self.heads
            qk_shape = (QUERY_KEY_HEADS, self.query_dim)
            if self.mlp_query_key:
                query = self.get(
                    "queryMLP",
                    MLP,
                    shape=None,
                    layers=2,
                    units=self.heads * self.query_dim,
                    act="silu",
                    **self.kw,
                )(query)
                # bs, Ne, heads, size
                query = self.get("query", LinearAct, qk_shape, norm="layer")(query)

                key = self.get(
                    "keyMLP",
                    MLP,
                    shape=None,
                    layers=2,
                    units=self.heads * self.key_dim,
                    act="silu",
                    **self.kw,
                )(key)
                # bs, Ne, heads, size
                key = self.get("key", LinearAct, qk_shape, norm="layer")(key)

            elif self.pos_head_seperate:
                # qk_shape = (QUERY_KEY_HEADS, self.query_dim)
                entity = query[:, :, :pos_dim]
                pos = query[:, :, pos_dim:]
                qk_dim = self.query_dim // self.heads

                query_entity = self.get(
                    "query_entity", LinearAct, (1, qk_dim), norm="layer"
                )(entity)
                query_pos = self.get("query_pos", LinearAct, (1, qk_dim), norm="layer")(
                    pos
                )
                # bs, Ne, heads=2, size
                query = jnp.concatenate([query_entity, query_pos], axis=-2)

                key_entity = self.get(
                    "key_entity", LinearAct, (1, qk_dim), norm="layer"
                )(entity)
                key_pos = self.get("key_pos", LinearAct, (1, qk_dim), norm="layer")(pos)
                key = jnp.concatenate([key_entity, key_pos], axis=-2)

                value_entity = self.get(
                    "value_entity", LinearAct, (1, self.value_dim), norm="layer"
                )(value_input)
                value_pos = self.get(
                    "value_pos", LinearAct, (1, self.value_dim), norm="layer"
                )(value_input)
                value = jnp.concatenate([value_entity, value_pos], axis=-2)

            else:
                # bs, Ne, heads, size
                query = self.get("query", LinearAct, qk_shape, norm="layer")(query)
                key = self.get("key", LinearAct, qk_shape, norm="layer")(key)
                value_shape = (self.heads, self.value_dim)
                value = self.get("value", LinearAct, value_shape)(value_input)

            # print shape of query, key, value
            print(
                f"attention.tensors | query_shape={query.shape} | "
                f"key_shape={key.shape} | value_shape={value.shape}"
            )

            # sum d: bs, N, h, d * bs, S, h, d -> bs, h, N, S
            logits = einsum("...Nhd,...Shd->...hNS", query, key)
            logits = logits / np.sqrt(self.query_dim // self.heads)

            if mask is not None:
                mask = mask[:, None, :, None]
                assert mask.ndim == logits.ndim
                logits = jnp.where(mask, logits, -INF)

            # bs, h, Ne, Ns
            atten_scores = jax.nn.softmax(logits)

        x = einsum("...hNS, ...Shd -> ...Nhd", atten_scores, value)
        x = rearrange(x, "... N h d -> ... N (h d)")
        assert x.ndim == 3, x.shape
        return x, atten_scores
