import re

# from ledwm.nets.Attention import Attention
from ledwm.nets.ImageEncoderResnet import ImageEncoderResnet
from ledwm.nets.embedding import Embedding

# from ledwm.nets.Initializer import Initializer
# from ledwm.embodied.envs.MessengerSent import split_state
from ledwm import ninjax as nj
import jax.numpy as jnp
from ledwm import nets

NUM_ENTITIES = 17
import jax.numpy as jnp


def nonzero_mean(emb):
    """
    Takes as input an embedding, emb. It should be H x W x L x D. with
    optional batch dimension. (H,W) is the grid dim, L the layers and
    D the embedding dimension. Returns mean of non-zero vectors along L dim.
    This is used to take care of overlapping sprites.
    """
    # Calculate the L2 norm along the last dimension
    norm = jnp.linalg.norm(emb, axis=-1)

    # Count the number of non-zero vectors along the L dimension
    non_zero = jnp.sum(norm > 0, axis=-2, keepdims=True).astype(jnp.float32)

    # Replace zeros with ones to prevent division by zero
    non_zero = jnp.where(non_zero == 0, 1, non_zero)

    # Sum along the L dimension and divide by the count of non-zero vectors
    return jnp.sum(emb, axis=-2) / non_zero


class EncoderHist(nj.Module):
    def __init__(
        self,
        shapes,  # obs_shapes
        embed_dim,
        winit="normal",
        atten={"heads": 1, "size": 256},
        cnn={"cnn_depth": 4, "cnn_blocks": 2, "resize": 16},
        **kw,
    ):
        excluded = ("is_first", "is_last")
        shapes = {
            k: v
            for k, v in shapes.items()
            if (k not in excluded and not k.startswith("log_"))
        }
        used_shape_keys = ["image", "state", "sent_embed"]
        used_shapes = {k: v for k, v in shapes.items() if k in used_shape_keys}
        print(f"sentence_history_encoder.inputs | shapes={used_shapes}")

        self.shapes = shapes
        self.preprocessors = {}
        self.winit = winit
        self.embed_dim = embed_dim
        self._kw_atten = atten
        self._cnn = ImageEncoderResnet(**cnn, name="cnn")

    def __call__(self, data: dict):
        """
        data dict of ['action', 'state' (3, 16, 16, Ne+1), 'is_first', 'is_last', 'is_terminal', 'reset', 'reward', 'cont']
        """
        some_key, some_shape = list(self.shapes.items())[0]
        # bs, bl = batch_dims
        batch_dims = data[some_key].shape[: -len(some_shape)]  # (bs, bl)

        # collapse bs, bl -> bs * bl: {bs * bl, -1} or (bs)
        # BS, BL = batch_dims
        data = {
            k: v.reshape((-1,) + v.shape[len(batch_dims) :]) for k, v in data.items()
        }

        K, H, W, _ = data["state"].shape[-4:]
        # bs * bl, T=3, 16, 16, L=[1,3,5]
        entity_state, avatar_state = jnp.split(
            data["state"], [data["state"].shape[-1] - 1], axis=-1
        )

        # bs * bl, 3, 16, 16, L, d
        entity_embed = self.get(
            "embedding_matrix",
            Initializer(self.winit),
            (NUM_ENTITIES, self.embed_dim),
        )[entity_state]
        # bs * bl, 3, 16, 16, d
        query = nonzero_mean(entity_embed)
        # bs * bl, 3*16*16, d
        query = query.reshape(query.shape[0], -1, query.shape[-1])

        # bs * bl, 3, 16, 16, 1, d
        avatar_embed = self.get("embedding_matrix")[avatar_state]
        # bs * bl, 3, 16, 16, d
        avatar = nonzero_mean(avatar_embed)

        # query = bs * bl, 3 * 16 * 16, d
        # key = bs * bl, S, d
        # value = bs * bl, S, d
        # state = bs * bl, 3 * 16 * 16, d
        state = self.get("atten", Attention, **self._kw_atten)(
            query, data["sent_embed"], data["sent_embed"]
        )
        state = state.reshape(state.shape[0], K, H, W, -1)
        state = (state + avatar) / 2.0

        # bs * bl, 3, 16, 16, d + L + 1
        inputs = jnp.concatenate([state, data["state"]], -1)
        # bs *bl, 16, 16, 3 * (d + L + 1)
        inputs = inputs.reshape(inputs.shape[0], H, W, -1)
        output = self._cnn.__call__(inputs)  # (bs*bl, cnn_dim=4*4*96*2=3092)
        output = output.reshape(batch_dims + output.shape[1:])  # (bs, bl, 3072)
        return output

        # for k, v in obs.items():
        #     obs[k] = v.reshape(batch_dims + v.shape[1:])
        # encoder_shapes = {k: v.shape for k, v in obs.items()}
        # print("Encoder shapes:", encoder_shapes)

        # return obs
