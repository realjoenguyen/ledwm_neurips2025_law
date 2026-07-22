from termcolor import cprint
import flax.linen as nn
import jax
from ledwm import ninjax as nj
import jax.numpy as jnp

from ledwm.jaxutils import apply_dropout_on

NUM_ALL_ENTITIES = 17
DEALTH_ENTITY = 17  # NUM_ENTITIES + 1 -> 0 - 17


class EncoderEmbed(nj.Module):
    def __init__(
        self,
        shapes,  # obs_shapes
        task,  # s1, s2, s3
        env_cache=None,
        use_sent_ids=False,
        embed_dim=32,
        winit="normal",
        read=False,
        mlp=None,
        use_time_step=False,
        # max_timestep=-1,
        small_image=False,
        use_lang=True,
        entity_aug=False,
        dropout=0,
        entity_track=False,
        config=None,
        use_movement_class=True,
        **kw,
    ):
        excluded = ("is_first", "is_last")
        self.config = config
        assert self.config is not None, self.config
        self.task = task
        self.use_sent_ids = use_sent_ids
        self.entity_track = entity_track
        self.entity_aug = entity_aug
        self.use_time_step = use_time_step
        self.small_image = small_image
        self.use_lang = use_lang
        shapes = {
            k: v
            for k, v in shapes.items()
            if (k not in excluded and not k.startswith("log_"))
        }
        used_shape_keys = [
            "image",
            "entity_ids",
            "avatar_ids",
            "entity_pos",
            "avatar_pos",
            "sent_embed" if use_sent_ids else "sent_ids",
        ]
        used_shapes = {k: v for k, v in shapes.items() if k in used_shape_keys}
        print(f"sentence_encoder.inputs | shapes={used_shapes}")
        self.entity_embed = nj.FlaxModule(
            nn.Embed, NUM_ALL_ENTITIES + 1, embed_dim, name="embed"
        )  # add DEATH_ENTITY

        # TODO debug
        self.use_movement_class = use_movement_class
        if self.use_movement_class:
            self.movement_embed = nj.FlaxModule(
                nn.Embed, 4, embed_dim, name="movement_embed"
            )

        # if self.config.use_role:
        self.role_embed = nj.FlaxModule(
            nn.Embed, NUM_ALL_ENTITIES + 1, embed_dim, name="role_embed"
        )

        self.read = read
        if self.read:
            self.read_embed = nj.FlaxModule(nn.Embed, 2, embed_dim, name="read_embed")

        if self.use_time_step:
            from ledwm.embodied.envs.MessengerSent import NUM_TIME_STEPS

            self.time_embed = nj.FlaxModule(
                nn.Embed, NUM_TIME_STEPS[task], embed_dim, name="time_embed"
            )

        # create jax array from numpy array env_cache
        if self.use_sent_ids:
            assert env_cache is not None
            num_sents = env_cache["sent_embed"].shape[0]
            sent_dim = env_cache["sent_embed"].shape[1]
            with jax.transfer_guard("allow"):
                self.sent_embed = nj.FlaxModule(
                    nn.Embed,
                    num_sents,
                    sent_dim,
                    None,
                    jnp.float32,
                    lambda *args: jnp.array(env_cache["sent_embed"]),
                    name="sent_embed",
                )
                self.sent_drop = nj.FlaxModule(
                    nn.Dropout,
                    rate=dropout,
                    name="sent_drop",
                )
            cprint("sentence_encoder.embeddings | source=env_cache", "green")

        self.shapes = shapes
        self.preprocessors = {}
        self.winit = winit
        self.embed_dim = embed_dim

    def __call__(self, data: dict, step=None, training=True):
        """
        data dict of ['action', 'image', 'is_first', 'is_last', 'is_read_step', 'is_terminal', 'reset', 'reward', 'token', 'token_embed', 'cont']
        Returns:
            {'entity_embed',
             'avatar_embed',
             'sent_embed'
            }
        """
        # collapse bs, bl -> bs * bl
        # some_key = image, some_shape = (16, 16, 17)
        some_key, some_shape = list(self.shapes.items())[0]
        # (bs, bl). Can be bs or (bs, bl)
        batch_dims = data[some_key].shape[: -len(some_shape)]
        # bs, bl, -1
        bs = batch_dims[0]
        data = {
            k: v.reshape((-1,) + v.shape[len(batch_dims) :]) for k, v in data.items()
        }

        # entity_ids: bs, bl, Ne
        entity_embed = self.entity_embed(data["entity_ids"])
        avatar_embed = self.entity_embed(data["avatar_ids"])
        assert entity_embed.shape[-1] == avatar_embed.shape[-1] == self.embed_dim, (
            f"{entity_embed.shape[-1]=} != {avatar_embed.shape[-1]=} != {self.embed_dim=}"
        )

        obs = {
            "entity_embed": entity_embed,
            "entity_pos": data["entity_pos"],
            "avatar_embed": avatar_embed,
            "avatar_pos": data["avatar_pos"],
        }

        if self.use_time_step:
            time_embed = self.time_embed(data["time_step"])
            obs["time_embed"] = time_embed
            obs["time_step"] = data["time_step"]
            if "read_embed" in obs:
                raise NotImplementedError

        if self.use_lang:
            if self.use_sent_ids:
                sent_embed = self.sent_embed(
                    data["sent_ids"]
                )  # sent_ids: bs*bl/envs(for policy), Ne
                sent_embed = apply_dropout_on(
                    sent_embed, self.sent_drop, training, step
                )
            else:
                sent_embed = data["sent_embed"]
            obs["sent_embed"] = sent_embed

        if self.entity_track:
            obs["dp"] = data["dp"]
            if self.use_movement_class and "movement" in data:
                data["movement"] = data["movement"].astype(jnp.int32)
                obs["movement_embed"] = self.movement_embed(data["movement"])

        if self.config.use_role:
            obs["role_embed"] = self.role_embed(data["role"].astype(jnp.int32))

        # IMAGE
        if "small_image" in data or "image" in data:
            obs["image"] = data["small_image"] if self.small_image else data["image"]

        assert self.config is not None
        if self.config.rssm.atten.gt_grounding:
            obs["gt_grounding_scores"] = data["manual_ids"]

        for k, v in obs.items():
            obs[k] = v.reshape(batch_dims + v.shape[1:])

        if self.config.drop_x_randomly_in_query and training:
            obs["drop_x"] = jax.random.bernoulli(nj.rng(), 0.5, (bs,))
            # extend to bl: from bs -> bs, bl
            if len(batch_dims) > 1:
                obs["drop_x"] = obs["drop_x"][:, None].repeat(batch_dims[1], axis=1)

        encoder_shapes = {k: v.shape for k, v in obs.items()}
        print(f"sentence_encoder.outputs | shapes={encoder_shapes}")

        return obs
