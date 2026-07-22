from einops import rearrange, repeat
from termcolor import cprint
from ledwm import ninjax as nj
from jax import numpy as jnp
from ledwm.nets.Attention import Attention
from ledwm.nets.ImageEncoderResnet import ImageEncoderResnet
from ledwm.nets.Linear import LinearAct
from ledwm.nets.MLP import MLP
from ledwm.embodied.envs.LWMSent import MIN_HIST_LEN
from ledwm.jaxutils import (
    apply_dropout_on,
    cast_to_compute,
    put_atten_to_grid,
    get_dist_z_from,
    sample_z_from,
    put_atten_to_grid_seperate,
)
import jax
import flax.linen as nn


class ZGenerator(nj.Module):
    def __init__(self, impl, stoch, classes, unimix, config, **kwargs):
        self.impl = impl
        self.stoch = stoch
        self.classes = classes
        self.unimix = unimix
        self.config = config
        self.kw = kwargs

    def __call__(self, x):
        if self.impl == "gaussian":
            raise NotImplementedError
            # x = self.get("mlp_input",LinearAct, 2 * self.stoch)(x)
            # print("Linear x:", x.shape)
            # # bs, bl, 2 * d_stoch=32
            # mean, std = jnp.split(x, 2, -1)
            # std = jax.nn.softplus(std) + 0.1
            # return {"mean": mean, "std": std}

        elif self.impl == "softmax":
            # bs, d_unit -> bs, d_stoch * n_class -> bs, d_stoch, n_class -> softmax -> mix with uniform -> logit: bs, d_stoch, n_class
            # linear only
            x = self.get(
                "mlp_input",
                # LinearAct,
                MLP,
                shape=None,
                layers=2,
                units=self.stoch * self.classes,
                act="silu",
                norm=self.kw["norm"],
            )(x)

            # bs, d_stoch, n_class
            logit = x.reshape(x.shape[:-1] + (self.stoch, self.classes))
            if self.unimix > 0:
                probs = jax.nn.softmax(logit, -1)  # prob: bs, d_stoch, n_class
                uniform = jnp.ones_like(probs) / probs.shape[-1]
                # mix with uniform
                probs = (1 - self.unimix) * probs + self.unimix * uniform
                logit = jnp.log(probs)

            return {"logit": logit}

        else:
            raise NotImplementedError


ATTEN_WEIGHT_KEY = "enc_atten"


class EncoderRSSM(nj.Module):
    """
    given embeddings, do Attention -> CNN
    """

    def __init__(
        self,
        entity_track_atten,
        config,
        cnn_atten: dict = None,
        cnn_image: dict = None,
        **kwargs,
    ):
        atten_kw = {
            **config.rssm.atten,
            "name": ATTEN_WEIGHT_KEY,
            "entity_track_atten": entity_track_atten,
        }
        print(f"encoder_rssm.config | component=language_attention | options={atten_kw}")
        self._attention = Attention(**atten_kw)
        self.init_cnn(cnn_atten, cnn_image, config, **kwargs)
        self.units = kwargs["units"]
        self.act = kwargs["act"]
        self.norm = kwargs["norm"]
        self.config = config
        self.obs_z = ZGenerator(
            impl=config.rssm.impl,
            stoch=config.rssm.stoch,
            classes=config.rssm.classes,
            unimix=config.rssm.unimix,
            config=config,
            name="obs_z",
            **kwargs,
        )
        if config.rssm.dropout > 0:
            self.dropout_func = nj.FlaxModule(
                nn.Dropout, rate=config.rssm.dropout, name="dropout"
            )
        self.kw = kwargs

    def init_cnn(self, cnn_atten, cnn_image, config, **kwargs):
        if config.rssm.cnn_atten is not None:
            cprint("encoder_rssm.config | cnn_attention=true", "green")
            # if self.use_atten:
            # cnn_depth = config.rssm.cnn_atten["cnn_depth"]
            # del cnn_atten["cnn_depth"]
            cnn_depth = cnn_atten.pop("cnn_depth", None)
            cnn_kw = {
                "depth": cnn_depth,
                "normalize": False,
                "concat_image": config.rssm.concat_image,
                **cnn_atten,
                **kwargs,
                "name": "cnn_attn",
            }
            cnn_kw.pop("units", None)
            print(f"encoder_rssm.config | component=attention_cnn | options={cnn_kw}")
            self.cnn_kw = cnn_kw
            self.cnn_atten = ImageEncoderResnet(**cnn_kw)

            if config.encode_image and config.rssm.seperate_cnn_image:
                cprint("encoder_rssm.config | separate_image_cnn=true", "green")
                assert config.rssm.cnn_image is not None
                cnn_image_kw = {
                    "normalize": False,
                    "concat_image": config.rssm.concat_image,
                    **cnn_image,
                    **kwargs,
                    "name": "cnn_image",
                }
                del cnn_image_kw["units"]
                print(
                    f"encoder_rssm.config | component=image_cnn | "
                    f"options={cnn_image}"
                )
                self.cnn_image = ImageEncoderResnet(**cnn_image_kw)

    def pos_input(self, data: dict):
        # entity_pos_hist: bs, Ne, 2 * hist
        if self.config.use_movement_class:
            raw_pos_input = data["movement_embed"]
        else:
            assert data["dp"] is not None, data["dp"]
            raw_pos_input = data["dp"]

        # bs, Ne, pos_dim
        if self.config.posMLP:
            # raw_pos_input = apply_dropout_on(
            #     raw_pos_input, self._dropout, training, step
            # )
            return self.get(
                "posMLP",
                MLP,
                shape=None,
                layers=2,
                units=self.config.rssm.pos_dim,
                act="silu",
            )(raw_pos_input)

        else:
            return raw_pos_input

    def drop_x_if_needed(self, data: dict, training=True):
        if (
            self.config.drop_x_randomly_in_query
            and training
            and data["drop_x"] is not None
        ):
            assert self.config.rssm.task in ["s2", "s3"], self.config.rssm.task
            cprint("encoder_rssm.query_dropout | enabled=true", "red")
            return jnp.where(data["drop_x"][:, None, None] > 0, 0, data["entity_embed"])
        else:
            return data["entity_embed"]

    def crop_image_to_rssm_grid(self, data: dict):
        if "image" in data:
            print(
                f"encoder_rssm.image_crop | shape={self.config.rssm.image_shape}"
            )
            data["image"] = data["image"][
                :,
                : self.config.rssm.image_shape[0],
                : self.config.rssm.image_shape[1],
                :,
            ]
        return data

    def atten_grid(
        self, input_atten, data: dict, training=True, step=None, verbose=False
    ):
        assert input_atten is not None
        assert data["sent_embed"] is not None
        print(
            f"encoder_rssm.attention | query_shape={input_atten.shape} | "
            f"key_shape={data['sent_embed'].shape} | "
            f"value_shape={data['sent_embed'].shape}"
        )

        if self.config.entity_track_and:
            raise NotImplementedError

        if self.config.pos_only and not self.config.use_movement_class:
            # raise NotImplementedError
            all_zeros = jnp.all(self.pos_input(data) == 0, axis=-1)  # bs, Ne
            assert data["time_step"] is not None
            # bs, Ne
            mask = jnp.where(
                jnp.logical_and(all_zeros, data["time_step"][:, None] <= MIN_HIST_LEN),
                0,
                1,
            )

        else:
            mask = None

        if self._attention.gt_grounding:
            assert data["gt_grounding_scores"] is not None

        # bs, Ne, d_atten or bs, Ne * hist, d_atten
        atten_out, atten_scores = self._attention.__call__(
            input_atten,  # bs, Ne, d_embed or d_deter + d_embed,
            # apply_dropout_on(sent_embed, self._dropout, training, step),
            # apply_dropout_on(sent_embed, self._dropout, training, step),
            data["sent_embed"],
            data["sent_embed"],
            mask=mask,
            training=training,
            step=step,
            gt_grounding_scores=data.get("gt_grounding_scores", None),
            pos_dim=self.config.rssm.pos_dim,
        )
        atten_out = cast_to_compute(atten_out)
        info = {}
        if verbose:
            info["atten"] = atten_scores

        # put back to entity_pos
        # (bs, Ne, 3) -> (bs, 16, 16, d_atten=256)
        if data["entity_embed"].shape[-1] != atten_out.shape[-1]:
            print(
                f"encoder_rssm.projection | input=avatar_embedding | "
                f"output_dim={self._attention.value_dim * self._attention.heads}"
            )
            avatar_embed = self.get(
                "avatar",
                LinearAct,
                self._attention.value_dim * self._attention.heads,
            )(data["avatar_embed"])  # bs, 1, d_dim or bs, hist, 1, d_dim
        else:
            avatar_embed = data["avatar_embed"]

        # bs, 10, 10, d_dim or bs * hist, 10, 10, d_dim
        attn_grid = put_atten_to_grid(
            data["entity_pos"],  #  bs, Ne, 3
            atten_out,  # bs, Ne, d_atten
            data["avatar_pos"],  # bs, 1, 3
            avatar_embed,  # bs, 1, d_avatar
            grid_size=tuple(self.config.rssm.image_shape),
        )
        return attn_grid, info

    def movement_role_lang_grid(self, data: dict):
        # concat role embed (as entity embed) and movement embed
        # assert self.config.env.lwm.use_role, "use_role must be True"
        if self.config.concat_role_movement:
            entity_embed = jnp.concatenate(
                [data["role_embed"], data["movement_embed"]], -1
            )
        else:
            entity_embed = data["role_embed"] + data["movement_embed"]
        if data["avatar_embed"].shape[-1] != entity_embed.shape[-1]:
            print(
                f"encoder_rssm.projection | input=avatar_embedding | "
                f"output_dim={entity_embed.shape[-1]}"
            )
            avatar_embed = self.get(
                "avatar",
                LinearAct,
                entity_embed.shape[-1],
            )(data["avatar_embed"])  # bs, 1, d_dim or bs, hist, 1, d_dim
        else:
            avatar_embed = data["avatar_embed"]

        if self.config.seperate_object_z:
            # bs, Ne, 16, 16, d_entity
            lang_grid = put_atten_to_grid_seperate(
                data["entity_pos"],  #  bs, Ne, 3
                entity_embed,  # bs, Ne, d_atten or bs, Ne * hist, d_atten
                data["avatar_pos"],  # bs, 1, 3
                avatar_embed,  # bs, 1, d_avatar
                grid_size=tuple(self.config.rssm.image_shape),
            )
        else:
            lang_grid = put_atten_to_grid(
                data["entity_pos"],  #  bs, Ne, 3
                entity_embed,  # bs, Ne, d_atten or bs, Ne * hist, d_atten
                data["avatar_pos"],  # bs, 1, 3
                avatar_embed,  # bs, 1, d_avatar
                grid_size=tuple(self.config.rssm.image_shape),
            )
        info = {}
        return lang_grid, info

    def __call__(
        self,
        data: dict,
        deter,  # infered history after data['action'] lead to this obs
        step=None,
        training=True,
        verbose=False,
    ):
        data = self.crop_image_to_rssm_grid(data)

        if self.config.rssm.entity_track_atten:
            if self.config.pos_only:
                input_atten = self.pos_input(data)
            else:
                # bs, Ne, dim
                input_atten = jnp.concatenate(
                    [self.drop_x_if_needed(data, training), self.pos_input(data)], -1
                )

        else:
            input_atten = data["entity_embed"]

        if self.config.movement_role_lang_grid:
            lang_grid, info = self.movement_role_lang_grid(data)
        else:
            lang_grid, info = self.atten_grid(
                input_atten, data, training, step, verbose
            )

        print(f"encoder_rssm.tensor | name=attention_grid | shape={lang_grid.shape}")

        # Handle seperate_object_z case where lang_grid is 5D
        if self.config.seperate_object_z:
            bs, Ne = lang_grid.shape[:2]
            spatial_dims = lang_grid.shape[2:]  # (16, 16, d_entity)
            # Reshape to 4D for CNN: (bs*Ne, 16, 16, d_entity)
            lang_grid_reshaped = lang_grid.reshape((bs * Ne, *spatial_dims))
        else:
            lang_grid_reshaped = lang_grid
            bs, Ne = lang_grid.shape[0], 1  # For consistency in reshape later

        if self.config.rssm.concat_image:
            # cnn (bs, 16, 16, d_atten) -> (bs, d_cnn)
            assert data["image"] is not None
            cnn_in = jnp.concatenate([lang_grid, data["image"]], -1)
            cnn_out = self.cnn_atten.__call__(
                cnn_in, data["image"].shape[-1], training=training
            )
            # (bs, d_cnn=4*4*96*2=)

        elif self.config.rssm.seperate_cnn_image:
            print("encoder_rssm.config | separate_cnns=true")
            # atten_grid = apply_dropout_on(attn_grid, self._drop, training, step)
            assert data["image"] is not None
            cnn_atten_out = self.cnn_atten.__call__(
                lang_grid, data["image"].shape[-1], training=training
            )
            cnn_image_out = self.cnn_image.__call__(data["image"], training=training)
            cnn_out = jnp.concatenate([cnn_atten_out, cnn_image_out], -1)

        else:
            cnn_out = self.cnn_atten.__call__(lang_grid_reshaped, training=training)

        # Reshape CNN output back if using seperate_object_z
        if self.config.seperate_object_z:
            # cnn_out shape: (bs*Ne, d_cnn) -> (bs, Ne, d_cnn)
            cnn_out = cnn_out.reshape((bs, Ne, cnn_out.shape[-1]))

        # else:
        #     assert data['image'] is not None
        #     batch_dims = data['image'].shape[:-3]
        #     assert len(batch_dims) == 1
        #     # random shuffle the last dim of image
        #     cnn_out = self.cnn_image.__call__(data['image'], training=training)

        if self.config.time_in == "obs":
            assert data["time_embed"] is not None
            if self.config.seperate_object_z:
                time_embed = repeat(data["time_embed"], "bs d -> bs Ne d", Ne=Ne)
            else:
                time_embed = data["time_embed"]

            input_z = jnp.concatenate([cnn_out, time_embed], -1)
            print("encoder_rssm.concat | input=time_embedding")

        print(f"encoder_rssm.tensor | name=cnn_output | shape={cnn_out.shape}")

        if self.config.rssm.concat_deter:
            # h_t, x_t, l_t - encoder - post
            if self.config.rssm.dropout > 0:
                deter = apply_dropout_on(deter, self.dropout_func, training, step)

            if self.config.seperate_object_z:
                deter = repeat(deter, "bs d -> bs Ne d", Ne=Ne)

            input_z = jnp.concatenate([deter, input_z], -1)
            print(
                f"encoder_rssm.concat | input=deterministic_state | "
                f"output_shape={input_z.shape}"
            )

        if self.config.residual_z:
            raise NotImplementedError
        else:
            # MLP 1 layer: [deter, z]
            encoded_obs = self.get(
                "obs_out",
                LinearAct,
                units=self.units,
                act=self.act,
                norm=self.norm,
            )(input_z)

        # if not self.concat_deter and not self.time:
        print(
            f"encoder_rssm.tensor | name=stochastic_input | "
            f"shape={encoded_obs.shape}"
        )
        stats = self.obs_z.__call__(encoded_obs)  # return logit
        z_dist = get_dist_z_from(stats, self.config.rssm.impl, step)
        stoch = sample_z_from(z_dist, argmax=self.config.z_argmax_train)

        if self.config.seperate_object_z:
            stoch = rearrange(stoch, "bs Ne d_stoch d_class -> bs (Ne d_stoch) d_class")
            stats["logit"] = rearrange(
                stats["logit"], "bs Ne d_stoch d_class -> bs (Ne d_stoch) d_class"
            )

        if self.config.time_in == "z":
            raise NotImplementedError

        # Keep posterior leaves in compute precision at their producer boundary
        # so the recurrent scan does not need a full-tree conversion each step.
        return (
            cast_to_compute(stats),
            cast_to_compute(stoch),
            cast_to_compute(encoded_obs),
            cast_to_compute(info),
        )
