import os
from collections import OrderedDict
from typing import Tuple, TypedDict
from einops import rearrange
from jax import numpy as jnp
from termcolor import cprint
from ledwm import (
    RSSM,
    CriticWM,
    jaxutils,
    ninjax as nj,
)
from ledwm.nets import (
    MLP,
    DecoderSent,
    EncoderSent,
)
from ledwm.nets.Dist import get_cubic_ease_in, get_linear_increase_kl, get_unimix_decay
import ledwm.nets.MultiDecoder
import ledwm.nets.MultiEncoder
import ledwm.Optimizer

from ledwm.nets.residual import ResidualDist, ResidualMLP
from ledwm.jaxutils import (
    add_dummy_first_action,
    create_horizon_mask_from,
    extract_horizon_data_from,
    get_dist_z_from,
    get_task,
    onehot_to_float,
    sg,
    symbolic_to_multihot,
    tensorstats,
    tree_map,
)
from ledwm.jaxutils import cast_to_compute as cast
import jax

from ledwm.constants import NON_ROLLOUT_INPUTS, NON_ROLLOUT_LOSSES

MIN_PRIORITY_BASELINE = {
    "s1": 1.5,
    "s2": 3,
    "s3": 1,
    "easy": 1,
    "medium": 1,
    "hard": 1,
}

# NUM_REWARD_CLASSES = 5  # -2, -1, 0, 0.5, 1
MASK_VALUE_DIST = 1e6
REWARD_VALUES = {
    "s1": [-1, 0, 1],
    "messenger_s1": [-1, 0, 1],
    "s2": [-1, 0, 0.5, 1],
    "messenger_s2": [-1, 0, 0.5, 1],
    "s3": [-2, -1, 0, 0.5, 1],
    "messenger_s3": [-2, -1, 0, 0.5, 1],
    "easy": [-1, 0, 0.5, 1],
    "lwm_easy": [-1, 0, 0.5, 1],
    "medium": [-1, 0, 0.5, 1],
    "lwm_medium": [-1, 0, 0.5, 1],
    "hard": [-1, 0, 0.5, 1],
    "lwm_hard": [-1, 0, 0.5, 1],
}


def fast_train_metrics(
    data,
    priority_loss_per_bs,
    sup_loss_per_batch,
    rollout_loss_per_bs,
):
    metrics = OrderedDict()
    metrics["priority_loss_per_batch"] = priority_loss_per_bs
    metrics["sup_loss_per_batch"] = sup_loss_per_batch
    metrics["rollout_loss_per_batch"] = rollout_loss_per_bs
    if "sample_id" in data:
        metrics["sample_id"] = data["sample_id"]
    return metrics


def get_bs_bw_weights(
    data,  # last_reward: bs,
    last_reward_weights,  # dict of {reward: weight}
):
    """
    Args:
        data (_type_):
        last_reward_weights (_type_):

    Returns:
        (bs, )
    """
    reward_keys = jnp.array(list(last_reward_weights.keys()))
    reward_values = jnp.array(list(last_reward_weights.values())).squeeze(-1)  # bs,

    def get_weight(reward):
        index = jnp.where(reward_keys == reward, size=1, fill_value=-1)[0]
        return jnp.where(
            index == -1, jnp.array(0.0), reward_values[index]
        )  # default value is 0 is not found in reward_class_weights

    balanced_weight = jnp.squeeze(
        jax.vmap(get_weight)(data["reward_indicator"]), -1
    )  # bs,

    return balanced_weight


# typedict for annotation here
class DecoderType(TypedDict):
    decoder: DecoderSent.DecoderSent
    reward: MLP.MLP
    cont: MLP.MLP
    action: MLP.MLP
    value: CriticWM.VFunction


class WM(nj.Module):
    def __init__(
        self, obs_space, act_space, config, shapes, env_cache=None, reward_values=None
    ):
        self.task = get_task(config)
        self.obs_space = obs_space
        self.act_space = act_space["action"]
        self.config = config

        if self.config.reward_head.dist == "onehot":
            assert reward_values is not None, "Reward values is None"
            self.reward_values = (
                reward_values[0] if len(reward_values.shape) == 2 else reward_values
            )
        else:
            self.reward_values = None

        if config.encoder_type == "sent":
            self.encoder = EncoderSent.EncoderEmbed(
                shapes,
                self.task,
                env_cache,
                **config.encoder_sent,
                config=config,
                name="enc_sent",
            )
        elif config.encoder_type == "token":
            self.encoder = ledwm.nets.MultiEncoder.MultiEncoder(
                shapes, **config.encoder, name="enc"
            )
        else:
            raise NotImplementedError(config.encoder_type)

        if self.config.rssm_type == "rssm":
            self.rssm = RSSM.RSSM(
                **config.rssm,
                env_cache=env_cache,
                config=config,
                name="rssm",
            )

        elif self.config.rssm_type == "early":
            #     self.rssm = nets.EarlyRSSM(**config.early_rssm, name="rssm")
            raise NotImplementedError

        elif self.config.rssm_type == "token":
            #     self.rssm = nets.TokenRSSM(**config.token_rssm, name="rssm")
            raise NotImplementedError
        else:
            raise NotImplementedError(self.config.rssm_type)

        if config.decoder_type == "sent":
            if self.rssm.lang_image_recon:
                shapes["lang_image"] = (12, 12, 64)
            decoder = DecoderSent.DecoderSent(
                shapes, **config.decoder_sent, task=self.task, name="dec_sent"
            )
        elif config.decoder_type == "token":
            raise NotImplementedError(config.decoder_type)
            # decoder = ledwm.nets.MultiDecoder.MultiDecoder(
            #     shapes, **config.decoder, name="dec"
            # )
        elif config.decoder_type == "none":
            decoder = None
        else:
            raise NotImplementedError(config.decoder_type)

        if config.residual_decoder:
            self.decoder_heads = {
                "reward": ResidualDist(
                    (
                        len(
                            REWARD_VALUES[self.task],
                        )
                        if config.reward_head.dist == "onehot"
                        else ()
                    ),
                    **config.reward_head,
                    name="reward",
                ),
                "cont": ResidualDist(
                    (),
                    **config.cont_head,
                    name="cont",
                ),
            }
        else:
            self.decoder_heads = {
                "reward": MLP.MLP(
                    (
                        (len(REWARD_VALUES[self.task]),)
                        if config.reward_head.dist == "onehot"
                        else ()
                    ),
                    **config.reward_head,
                    name="rew",
                ),
                "cont": MLP.MLP((), **config.cont_head, name="cont"),
            }

        if decoder is not None:
            self.decoder_heads["decoder"] = decoder

        if self.config.action_pred:
            self.decoder_heads["prev_action"] = MLP.MLP(
                shape=self.act_space.shape,
                **config.action_head,
                name="act",
            )
        if self.config.value_pred:
            self.value_predictor = CriticWM.VFunction(
                config,
                name="value",  # type: ignore
            )
            self.decoder_heads["value"] = self.value_predictor.net

        self.opt = ledwm.Optimizer.Optimizer(
            name="model_opt", config=config, **config.model_opt
        )
        # self.opt_rollout = ledwm.Optimizer.Optimizer(
        #     name="rollout_opt", not_optimize=True, config=config, **config.model_opt
        # )
        scales = self.config.loss_scales.copy()
        image, vector = scales.pop("image"), scales.pop("vector")

        if decoder is not None:
            scales.update({k: image for k in self.decoder_heads["decoder"].cnn_shapes})
            scales.update({k: vector for k in self.decoder_heads["decoder"].mlp_shapes})

        self.scales = scales

    def initial(self, batch_size, use_for_policy=False):
        prev_latent = self.rssm.initial(batch_size, use_for_policy=use_for_policy)
        prev_action = jnp.zeros((batch_size, *self.act_space.shape))
        # if self.config.run.opt_step:
        #     return prev_latent, prev_action, 0
        # else:
        return prev_latent, prev_action

    def train(
        self,
        data: dict,
        state: dict,
        step=None,
        last_reward_weights=None,
    ) -> Tuple[dict, dict, dict]:
        for key in [x for x in self.config.zero_data_keys if x]:
            data[key] = jnp.zeros_like(data[key])
        modules = [
            self.encoder,
            self.rssm,
            *self.decoder_heads.values(),
        ]  # use to adjust skip_training

        if self.config.skip_mlp_training:
            raise NotImplementedError("skip_mlp_training not implemented")
        if self.config.skip_cnn_training:
            raise NotImplementedError("skip_cnn_training not implemented")

        mets, (state, outs, metrics) = self.opt.__call__(
            modules,
            self.loss,
            data,
            state,
            step,
            last_reward_weights,
            has_aux=True,
        )  # type: ignore

        if self.config.slow_encoder_fraction > 0:
            self.rssm.encoder_updater.__call__()

        # rollout_mets, (state, outs, metrics) = self.opt_rollout.__call__(
        #     modules,
        #     self.loss,
        #     data,
        #     state,
        #     step,
        #     last_reward_weights,
        #     rollout_only=True,
        #     has_aux=True,
        # )  # type: ignore

        # state: ({'deter', 'stoch', 'logits'}, (bs, bl))
        # outs: all losses + 'prior' + 'post'
        assert isinstance(metrics, dict), metrics
        assert isinstance(mets, dict), mets
        assert isinstance(outs, dict), outs

        metrics.update(mets)
        # metrics.update(rollout_mets)

        return state, outs, metrics

    def check_use_opt_step(self):
        return self.config.kl_weight != "none" or self.config.rssm.soft_z

    def rollout_one_step_image_loss(
        self,
        data,
        post,
        step=None,
        training=True,
    ):
        bs, bl = data["action"].shape[:2]
        cur_feat = jnp.concatenate(
            [
                data["action"],  # next action given cur obs, bs, bl, 5
                post["stoch"].reshape(bs, bl, -1),  # bs, bl, stoch*class
            ],
            -1,
        )  # bs, bl, 32*32 + 5

        deter = self.get(
            "deter_fwd_image",
            MLP.MLP,
            shape=None,
            layers=3,
            units=self.rssm.kw["units"],
            act="silu",
        )(cur_feat)

        next_image_dist = {}
        deter_input = {"deter": deter}

        out = self.decoder_heads["decoder"].__call__(deter_input)
        out = out if isinstance(out, dict) else {"decoder": out}
        # change key name from image to next_image
        out["next_image"] = out.pop("image")
        next_image_dist.update(out)

        # loss
        next_image_loss = {}
        # bs, bl, H=1
        # shift data['image'] by 1: bs, bl, 12, 12, 17
        next_image = jnp.concatenate(
            [data["image"][:, 1:], jnp.zeros((bs, 1, 12, 12, 17))], axis=1
        )
        assert next_image.shape == (bs, bl, 12, 12, 17), next_image.shape
        # remove dim 1
        loss = -next_image_dist["next_image"].log_prob(next_image)
        assert loss.shape == (bs, bl), loss.shape
        # remove loss at the end of the horizon
        loss = loss * (1 - data["is_last"])
        next_image_loss["next_image"] = loss

        return next_image_dist, next_image_loss

    def rollout_multi_step_loss(
        self,
        data,
        post,
        obs_input,
        step=None,
        rollout_dyn_free=1,
        rollout_rep_free=1,
        training=True,
    ):
        # start = {k: v[:, 0] for k, v in post.items()}  # bs, d*
        # assert "start_id" in data, "start_id not in data"
        # assert "end_id" in data, "end_id not in data"

        rollout_losses = {}
        bs, bl = data["action"].shape[:2]
        horizon = self.config.imag_horizon
        horizon_mask_from_next = create_horizon_mask_from(
            data["is_first"], data["is_last"], horizon, include_current_step=False
        )  # bs, bl, H
        horizon_mask_from_cur = create_horizon_mask_from(
            data["is_first"], data["is_last"], horizon, include_current_step=True
        )

        # jax.debug.print("horizon_mask {x}", x=horizon_mask)
        start = {
            k: v.reshape(-1, *(v.shape[2:]))
            for k, v in post.items()
            if k not in NON_ROLLOUT_INPUTS
        }  # bs * bl, *d

        # jax debug print
        # jax.debug.print("data['time_embed'] {x}", x=data["time_embed"])
        # CHECK time_embed can be updated

        rollout_action_from_cur = extract_horizon_data_from(
            data["action"],
            horizon,
            mask=horizon_mask_from_cur,
            include_current_step=True,
        )  # bs, bl, H, *d
        rollout_action_from_cur = rollout_action_from_cur.reshape(
            -1, *rollout_action_from_cur.shape[2:]
        )  # bs * bl, H, *d

        time_embed_start = obs_input["time_embed"].reshape(
            -1, *obs_input["time_embed"].shape[2:]
        )
        prior_imagine = self.rssm.imagine(
            cast(rollout_action_from_cur),  # bs * bl, H, 5
            cast(start),  # bs * bl, *d
            step=step,  # int
            cont_decoder=self.decoder_heads["cont"],
            training=training,
            time_embed_start=time_embed_start,
        )  # bs * bl, H, *d

        prior_imagine = {
            k: v.reshape(bs, bl, *v.shape[1:]) for k, v in prior_imagine.items()
        }  # bs, bl, H, *d

        rollout_dists = {}
        if self.config.rollout_image:
            # remove decoder from NON_ROLLOUT_LOSSES
            if "decoder" in NON_ROLLOUT_LOSSES:
                NON_ROLLOUT_LOSSES.remove("decoder")

        for name, head in self.decoder_heads.items():
            if name in NON_ROLLOUT_LOSSES:
                continue

            assert name in self.config.grad_heads, f"{name} not in grad_heads"
            out = head.__call__(prior_imagine)
            out = out if isinstance(out, dict) else {name: out}
            rollout_dists.update(out)

        for key, rollout_dist in rollout_dists.items():
            if key == "mean_sent_embed":
                # data['sent_embed']: bs, bl, s, d_s. mean over s
                gt_data = obs_input["sent_embed"].mean(2).astype(jnp.float32)

            elif key == "image" and self.config.small_image:
                gt_data = symbolic_to_multihot(data["small_image"]).astype(jnp.float32)

            elif key == "value":
                continue

            elif key == "lang_image":
                gt_data = post["lang_image"].astype(jnp.float32)

            else:
                gt_data = data[key].astype(jnp.float32)

            # bs, bl, horizon, d*
            mask_value_id = None
            if key == "reward" and self.config.reward_head.dist == "onehot":
                mask_value_id = 2  # [0, 0, 1, 0, 0] -> 0

            data_horizon_from_next = extract_horizon_data_from(
                gt_data,
                horizon,
                mask=horizon_mask_from_next,
                mask_value_id=mask_value_id,
            )
            loss = -rollout_dist.log_prob(data_horizon_from_next)  # bs, bl, horizon
            assert loss.shape == (bs, bl, self.config.imag_horizon), (key, loss.shape)

            if self.config.decay_multi_step:
                lambda_powers = jnp.power(
                    self.config.decay_lambda,
                    jnp.arange(1, self.config.imag_horizon + 1),
                )
                loss = loss * lambda_powers[None, None, :]  # bs, bl, horizon

            loss = loss * horizon_mask_from_next
            rollout_losses["rollout_" + key] = loss
            if key == "reward" and hasattr(rollout_dist, "kl"):
                rollout_losses["rollout_reward_kl"] = (
                    rollout_dist.kl(data_horizon_from_next) * horizon_mask_from_next
                )

        if self.config.value_pred:
            assert "value" in self.decoder_heads, "Value head not in decoder_heads"
            assert "value" in rollout_dists, "Value dist not in dists"
            combined_data_post = {**data, **post}
            # bs, bl, horizon, d*
            value_data_horizon = {
                k: extract_horizon_data_from(v, horizon, mask=horizon_mask_from_next)
                for k, v in combined_data_post.items()
                if k in CriticWM.VALUE_WM_KEYS
            }
            rollout_value_loss = self.value_predictor.critic_loss(value_data_horizon)[1]
            # add zero to the end of horizon - 1
            # bs, bl, horizon
            rollout_value_loss = jnp.concatenate(
                [rollout_value_loss, jnp.zeros((bs, 1, horizon))], 1
            )
            rollout_value_loss = rollout_value_loss * horizon_mask_from_next
            assert rollout_value_loss.shape == (bs, bl, self.config.imag_horizon), (
                rollout_value_loss.shape
            )

            if self.config.decay_multi_step:
                lambda_powers = jnp.power(
                    self.config.decay_lambda,
                    jnp.arange(1, self.config.imag_horizon + 1),
                )
                rollout_value_loss = rollout_value_loss * lambda_powers[None, None, :]

            rollout_losses["rollout_value"] = rollout_value_loss

        # add dyn loss here
        # if self.config.dyn_rep_multi_step:
        # into the future, NOT including the current timestep
        # bs, bl, H, d*
        post_horizon_from_next = {
            "logit": extract_horizon_data_from(
                post["logit"], horizon, mask=horizon_mask_from_next
            )
        }

        post_z_sg = get_dist_z_from(
            sg(post_horizon_from_next), self.config.rssm.impl, step
        )
        prior_z = get_dist_z_from(prior_imagine, self.config.rssm.impl, step)
        rollout_dyn_loss = post_z_sg.kl_divergence(prior_z)

        # if self.config.rssm_loss.multi_step_free > 0:
        #     rollout_dyn_loss = jnp.maximum(
        #         rollout_dyn_loss, self.config.rssm_loss.multi_step_free
        #     )

        # # add rep loss here
        post_z = get_dist_z_from(post_horizon_from_next, self.config.rssm.impl, step)
        prior_z_sg = get_dist_z_from(sg(prior_imagine), self.config.rssm.impl, step)
        rollout_rep_loss = post_z.kl_divergence(prior_z_sg)
        rollout_losses["rollout_dyn_kl"] = rollout_dyn_loss * horizon_mask_from_next
        rollout_losses["rollout_rep_kl"] = rollout_rep_loss * horizon_mask_from_next

        if self.config.dyn_rep_multi_step:
            if self.config.decay_multi_step:
                lambda_powers = jnp.power(
                    self.config.decay_lambda,
                    jnp.arange(1, self.config.imag_horizon + 1),
                )
                # bs, bl, horizon
                rollout_rep_loss = rollout_rep_loss * lambda_powers[None, None, :]

            if rollout_rep_free > 0:
                rollout_rep_loss = jnp.maximum(rollout_rep_loss, rollout_rep_free)

            if rollout_dyn_free > 0:
                rollout_dyn_loss = jnp.maximum(rollout_dyn_loss, rollout_dyn_free)

            if self.config.decay_multi_step:
                lambda_powers = jnp.power(
                    self.config.decay_lambda,
                    jnp.arange(1, self.config.imag_horizon + 1),
                )
                # bs, bl, horizon
                rollout_dyn_loss = rollout_dyn_loss * lambda_powers[None, None, :]
                # bs, bl, horizon
                rollout_rep_loss = rollout_rep_loss * lambda_powers[None, None, :]

            rollout_dists["rep"] = rollout_rep_loss
            # bs, bl, horizon
            rollout_losses["rollout_rep"] = rollout_rep_loss * horizon_mask_from_next

            if self.config.rep_anneal:
                # assert rollout_rep_free == 0.0 and rollout_dyn_free == 0.0
                if self.config.rep_anneal_params.decay == "linear":
                    rep_weight = get_linear_increase_kl(
                        self.config.rep_anneal_params.init,
                        self.config.rep_anneal_params.final,
                        self.config.rep_anneal_params.steps,
                        step,
                        step_init=self.config.rep_anneal_params.step_init,
                    )  # (bs, )
                elif self.config.rep_anneal_params.decay == "cubic":
                    rep_weight = get_cubic_ease_in(
                        self.config.rep_anneal_params.init,
                        self.config.rep_anneal_params.final,
                        self.config.rep_anneal_params.steps,
                        step,
                        step_init=self.config.rep_anneal_params.step_init,
                    )  # (bs, )
                # bs, bl, horizon
                else:
                    raise ValueError("Invalid decay type")

                rollout_rep_loss = rollout_rep_loss * rep_weight[:, None, None]

                if self.config.dyn_rep_anneal:
                    rollout_dyn_loss = rollout_dyn_loss * rep_weight[:, None, None]

                # rollout_rep_loss = jnp.where(
                #     step[0] < self.config.rep_anneal_params.step_init,
                #     jnp.maximum(rollout_rep_loss, rollout_rep_free),
                #     rollout_rep_loss,
                # )
                # rollout_dyn_loss = jnp.where(
                #     step[0] < self.config.rep_anneal_params.step_init,
                #     jnp.maximum(rollout_dyn_loss, rollout_dyn_free),
                #     rollout_dyn_loss,
                # )

            rollout_dists["dyn"] = rollout_dyn_loss
            rollout_losses["rollout_dyn"] = rollout_dyn_loss * horizon_mask_from_next
            rollout_losses["rollout_rep"] = rollout_rep_loss * horizon_mask_from_next

        if self.config.loss_up_is_first_weight > 0:
            cprint(
                f"model.rollout_loss_weight | condition=is_first | "
                f"weight={self.config.loss_up_is_first_weight}",
                "yellow",
            )
            for k, v in rollout_losses.items():
                assert "rollout_" in k, k
                assert v.shape == (bs, bl, self.config.imag_horizon), (k, v.shape)
                rollout_losses[k] = jnp.where(
                    rearrange(data["is_first"], "bs bl -> bs bl 1"),
                    v * self.config.loss_up_is_first_weight,
                    v,
                )

        # add rollout_to keys of rollout_dists
        # rollout_reward, rollout_cont
        rollout_dists = {f"rollout_{k}": v for k, v in rollout_dists.items()}

        cprint(
            f"model.distributions | scope=rollout | count={len(rollout_dists)}",
            "green",
        )
        for k, v in rollout_dists.items():
            print(f"model.distribution | scope=rollout | name={k} | value={v}")

        return rollout_losses, rollout_dists, horizon_mask_from_next

    def loss(
        self,
        data: dict,
        state: tuple,
        step=None,
        balanced_reward_weights=None,
        rollout_only=False,
    ):
        """
        data['action'][t]: next action given the obs[t]
        """

        # (bs, bl, cnn_dim + mlp_dim)
        if self.config.encoder_type == "token":
            raise NotImplementedError
        else:
            assert isinstance(self.encoder, EncoderSent.EncoderEmbed)
            obs_input: dict = self.encoder.__call__(data, step)

        # {'deter': (bs, d_deter=4096), 'stoch': (bs, 32, 32), 'logit': (bs, 32, 32)}; (num_act, bs)
        latent, action = state
        # (bs, bl, act)
        # actions = jnp.concatenate([action[:, None], data["action"][:, :-1]], 1) #

        if self.config.rssm_type == "token":
            raise NotImplementedError
        else:
            assert obs_input is not None, "Embedding is None"
            post = self.rssm.observe(
                obs_input,
                data["action"],
                data["is_first"],
                latent,
                step=step,  # in case of soft-z
                encoder_type=self.config.encoder_type,
            )

        dists = {}

        for name, head in self.decoder_heads.items():
            # inp = post_no_image if name in self.config.no_concat_image else post
            if name == "value":
                continue
            if self.config.one_step_image and name == "decoder":
                continue

            inp = post if name in self.config.grad_heads else sg(post)
            out = head.__call__(inp)
            out = out if isinstance(out, dict) else {name: out}
            dists.update(out)

        losses = {}
        if self.config.value_pred:
            dists["value"], losses["value"], value_metrics = (
                self.value_predictor.critic_loss({**post, **data})
            )
            bs, bl = data["action"].shape[:2]
            assert losses["value"].shape == (bs, bl - 1), losses["value"].shape
            # add zero to the end of bl -1 -> bl
            losses["value"] = jnp.concatenate([losses["value"], jnp.zeros((bs, 1))], 1)

        cprint(f"model.distributions | scope=supervised | count={len(dists)}", "green")
        for k, v in dists.items():
            print(f"model.distribution | scope=supervised | name={k} | value={v}")

        # given h_t after seeing z_t; find prior z^hat_t
        # rssm_losses:
        # dyn - prediction loss - train the z^hat_t from history h_t to match with the actual representation z_t from obs
        # regulize the actual representation z_t to match with the prior z^hat_t - predictable from history h_t
        rssm_losses, prior = self.rssm.loss(
            post,
            step,
            dyn_free=self.config.rssm_loss.dyn_free,
            rep_free=self.config.rssm_loss.rep_free,
            data=data,
        )
        losses.update(rssm_losses)  # dyn: (5, 256), rep: (5, 256)

        # token_emb, image, reward, cont loss
        bs, bl = data["action"].shape[:2]
        if self.config.action_pred:
            # create prev_action in data
            # prev_action[t]: action | obs[t-1] that leads to obs[t]
            prev_action = add_dummy_first_action(data["action"])
            data["prev_action"] = prev_action

        for key, dist in dists.items():
            if key == "mean_sent_embed":
                # data['sent_embed']: bs, bl, s, d_s. mean over s
                gt_data = obs_input["sent_embed"].mean(2).astype(jnp.float32)

            elif key == "image" and self.config.small_image:
                gt_data = symbolic_to_multihot(data["small_image"]).astype(jnp.float32)

            elif key == "lang_image":
                gt_data = post["lang_image"].astype(jnp.float32)

            elif key == "value":
                continue

            else:
                gt_data = data[key].astype(jnp.float32)

            # gt_data: image: bs, bl, 10, 10, 17
            loss = -dist.log_prob(gt_data)  # bs, bl
            assert loss.shape == (bs, bl), (key, loss.shape)

            if key == "prev_action":
                # action_pred: ignore the first timestep at every eps
                loss = loss * (1 - data["is_first"])

            losses[key] = loss
            if key == "reward" and hasattr(dist, "kl"):
                losses["reward_kl"] = dist.kl(gt_data)

        # add multi-step losses
        rollout_dists = {}
        horizon_mask_from_next = None
        if self.config.multi_step:
            rollout_losses, rollout_dists, horizon_mask_from_next = (
                self.rollout_multi_step_loss(
                    data,
                    post,
                    obs_input,
                    step,
                    rollout_dyn_free=self.config.rssm_loss.rollout_dyn_free,
                    rollout_rep_free=self.config.rssm_loss.rollout_rep_free,
                )
            )
            assert "rollout_reward" in rollout_losses, "No rollout_reward in losses"
            assert "rollout_cont" in rollout_losses, "No rollout_cont in losses"
            losses.update(rollout_losses)

        if self.config.one_step_image:
            next_image_dist, next_image_loss = self.rollout_one_step_image_loss(
                data, post, step
            )
            losses.update(next_image_loss)
            dists.update(next_image_dist)

        if self.config.rep_first is False:
            # dyn and rep at is_first = rep_first_min
            losses["dyn"] = jnp.where(
                data["is_first"], self.config.rep_first_min, losses["dyn"]
            )
            losses["rep"] = jnp.where(
                data["is_first"], self.config.rep_first_min, losses["rep"]
            )
            if "real_dyn" in losses:
                losses["real_dyn"] = jnp.where(
                    data["is_first"], self.config.rep_first_min, losses["real_dyn"]
                )

        scaled = {}
        for k, v in losses.items():
            if k in self.scales:
                if self.scales[k] > 0:
                    scaled[k] = v * self.scales[k]
            else:
                cprint(f"model.loss_scale_missing | name={k}", "red")

        supervised_losses = {k: v for k, v in scaled.items() if "rollout" not in k}
        cprint(
            f"model.losses | scope=supervised | count={len(supervised_losses)}",
            "green",
        )
        for k, v in supervised_losses.items():
            print(f"model.loss | scope=supervised | name={k} | shape={v.shape}")

        rollout_losses_opt = {k: v for k, v in scaled.items() if "rollout" in k}
        cprint(
            f"model.losses | scope=rollout | count={len(rollout_losses_opt)}",
            "green",
        )
        for k, v in rollout_losses_opt.items():
            print(f"model.loss | scope=rollout | name={k} | shape={v.shape}")

        priority_loss_per_bs = sum(supervised_losses.values()).mean(1)
        # assert priority_loss_per_bs no nan in jax
        # assert not jnp.isnan(priority_loss_per_bs).any(), priority_loss_per_bs

        rollout_loss_per_bs = 0
        if self.config.multi_step:
            assert len(rollout_losses_opt) > 0, "No rollout losses"
            rollout_losses_sum = sum(rollout_losses_opt.values())
            # bs, bl, horizon -> bs only --- sum over the last two dim
            horizon_mask_sum_per_bs = horizon_mask_from_next.reshape(bs, -1).sum(
                1
            ) * len(rollout_losses_opt)
            rollout_loss_per_bs = (
                rollout_losses_sum.reshape(bs, -1).sum(-1) / horizon_mask_sum_per_bs
            )
            priority_loss_per_bs += rollout_loss_per_bs

        priority_loss_per_bs = jnp.maximum(
            0, priority_loss_per_bs - MIN_PRIORITY_BASELINE[self.task]
        )

        if (
            self.config.replay.imbalance == "balanced_weight"
            and balanced_reward_weights is not None
        ):
            bs_bw = get_bs_bw_weights(data, balanced_reward_weights)
            assert bs_bw.shape == (bs,), bs_bw.shape

            supervised_losses = {
                k: v * bs_bw.reshape(bs, *([1] * len(v.shape[1:])))
                for k, v in supervised_losses.items()
            }
            rollout_losses_opt = {
                k: v * bs_bw.reshape(bs, *([1] * len(v.shape[1:])))
                for k, v in rollout_losses_opt.items()
            }

        supervised_loss = sum(supervised_losses.values())  # bs, bl
        assert isinstance(supervised_loss, jnp.ndarray), supervised_loss
        optimize_loss = supervised_loss.mean()  # bs, bl -> mean
        sup_loss_per_batch = supervised_loss.mean(1)

        if self.config.multi_step:
            rollout_losses_sum = sum(rollout_losses_opt.values())  # bs, bl, horizon
            # model_loss_mean += rollout_loss.mean()
            # mean over all horizon_mask
            assert isinstance(rollout_losses_sum, jnp.ndarray), rollout_losses_sum
            assert isinstance(horizon_mask_from_next, jnp.ndarray), (
                horizon_mask_from_next
            )
            # use multiple rollout_losses here: dyn, rep, reward, cont
            rollout_losses_mean = rollout_losses_sum.sum() / (
                horizon_mask_from_next.sum() * len(rollout_losses_opt)
            )
            optimize_loss += rollout_losses_mean
            if rollout_only:
                optimize_loss = rollout_losses_mean

        # out = {"embed": embed, "post": post, "prior": prior}
        out = {"post": post, "prior": prior}
        out.update({f"{k}_loss": v for k, v in losses.items()})
        last_latent = {
            k: v[:, -1] for k, v in self.rssm.recurrent_state(post).items()
        }
        last_action = data["action"][:, -1]

        state = last_latent, last_action
        if os.environ.get("LEDWM_FAST_TRAIN_METRICS") == "1":
            metrics = fast_train_metrics(
                data,
                priority_loss_per_bs,
                sup_loss_per_batch,
                rollout_loss_per_bs,
            )
        else:
            metrics = self._metrics(
                data,
                {**dists, **rollout_dists},
                post,
                prior,
                losses,  # not scale
                priority_loss_per_bs,
                sup_loss_per_batch,
                rollout_loss_per_bs,
                step,
            )
        if self.config.value_pred:
            metrics.update(value_metrics)

        return optimize_loss, (state, out, metrics)

    def imagine(
        self,
        policy,  # return {'action': action}, carry - ActorCritic.policy
        start,  # only need deter, logit, stoch
        horizon,
        carry=None,
        step=None,
        training=True,
    ):
        """
        imagination in horizon step + the first posterior step
        """
        # start: bs, *
        # carry = {}
        # policy: ActorCritic.policy
        if carry is None:

            def policy(s, c, f=policy):
                return (f(s), {})

            carry = {}

        state_keys = list(self.rssm.initial(1, use_for_policy=True).keys())
        state = {k: v for k, v in start.items() if k in state_keys}
        first_cont = (1.0 - start["is_terminal"]).astype(jnp.float32)
        # state['cont'] = first_cont
        if self.config.imag_cont_hard:
            carry["cont"] = first_cont

        # action: bs * bl, action_dim=5
        action, carry, info = policy(state, carry, step)
        keys = list(state.keys()) + list(action.keys()) + list(carry.keys())
        assert len(set(keys)) == len(keys), ("Colliding keys", keys)

        def step_fn(prev, timestep):
            prev_state, prev_action, carry = prev
            # get the prior from state + action
            # state: 'deter': bs, d_deter; 'logit': bs, n, H; 'stoch':  bs, n, H
            state = self.rssm.imagine_step(
                prev_state, prev_action["action"], step=step, training=training
            )
            action, carry, info = policy(state, carry, step)
            # if cont = 1, then take the cont from the model; else cont = 0, then keep the cont = 0
            # jax.debug.print("imagine policy = {probs}", probs=info["probs"])
            if self.config.imag_cont_hard:
                carry["cont"] = (
                    self.decoder_heads["cont"](state, training=training)
                    .mode()
                    .astype(jnp.float32)
                    * carry["cont"]
                )

            return state, action, carry

        # carries = {}
        # states: T, bs * bl, *
        # actions: T, bs * bl, *
        states, actions, carries = jaxutils.scan(
            step_fn,
            jnp.arange(horizon),  # inputs: (T, )
            (state, action, carry),  # start: (bs * bl, )
            self.config.imag_unroll,  # unroll default=False
        )
        # imagination over horizon step + 1: the initial step
        states, actions, carries = tree_map(
            lambda traj, first: jnp.concatenate([first[None], traj], 0),
            (states, actions, carries),  # traj
            (state, action, carry),  # first
        )

        # carries = {}
        # states: T, bs * bl, *
        # actions: T, bs * bl, *
        traj = {**states, **actions, **carries}

        if not self.config.imag_cont_hard:
            if self.config.imag_cont == "mode":
                cont = self.decoder_heads["cont"](traj).mode()
            # elif self.config.imag_cont == "mean":
            #     cont = self.decoder_heads["cont"](traj).mean()
            else:
                raise NotImplementedError(self.config.imag_cont)
            traj["cont"] = jnp.concatenate([first_cont[None], cont[1:]], 0)

        # because the traj doesn't have the first step - initial step
        discount = 1 - 1 / self.config.horizon
        traj["weight"] = jnp.cumprod(discount * traj["cont"], 0) / discount
        return traj

    def report(
        self,
        data,  # data: dict, each val with shape (min_bs, min_bl, <obs shape>)
        step=None,
        training=False,
    ):
        report = {}
        max_first_bs = self.config.run.report.first_bs
        max_first_bl = self.config.run.report.first_bl
        data = {
            k: v[:max_first_bs, :max_first_bl] if len(v.shape) > 1 else v
            for k, v in data.items()
        }
        N_FIRST_BS = data["is_first"].shape[0]
        N_FIRST_BL = data["is_first"].shape[1]
        state = self.initial(N_FIRST_BS)

        # take metrics from self.loss
        metrics_loss = self.loss(data, state, step)[-1][-1]
        report.update(metrics_loss)

        if self.config.rssm_type == "token":
            raise NotImplementedError
        else:
            if self.config.encoder_type == "token":
                raise ValueError("Token encoder not implemented")

            elif self.config.encoder_type == "sent":
                assert isinstance(self.encoder, EncoderSent.EncoderEmbed)
                obs_input = self.encoder.__call__(data, training=training)
                post = self.rssm.observe(
                    obs_input,
                    data["action"],
                    data["is_first"],
                    encoder_type="sent",
                    step=step,
                    training=training,
                    verbose=True,  # take atten score
                )
            else:
                raise NotImplementedError(self.config.encoder_type)

        # context:
        # - deter (bs, bl, rssm.deter)
        # - logit, stoch (bs, bl, rssm.stoch, rssm.classes)
        # posterior - observe data - take the last element [-1]
        # start = {k: v[:, 0] for k, v in context.items() if "atten" not in k}

        reward_pred = self.decoder_heads["reward"](post, training=training)
        cont_pred = self.decoder_heads["cont"](post, training=training)

        bs, bl = data["action"].shape[:2]
        # for cur_h in range(1, self.config.imag_horizon + 1):
        horizon = self.config.imag_horizon
        horizon_mask = create_horizon_mask_from(
            data["is_first"], data["is_last"], horizon
        )  # bs, bl, H

        future_horizon_mask_from_cur = create_horizon_mask_from(
            data["is_first"], data["is_last"], horizon, include_current_step=True
        )
        future_action_from_cur = extract_horizon_data_from(
            data["action"],
            horizon,
            mask=future_horizon_mask_from_cur,
            include_current_step=True,
        )  # bs, bl, H, *d

        rollout_data_reward = extract_horizon_data_from(
            data["reward"],
            horizon,
            mask=horizon_mask,
            mask_value_id=(
                REWARD_VALUES[self.config.task].index(0)
                if self.config.reward_head.dist == "onehot"
                else None
            ),
        )  # bs, bl, H

        rollout_data_cont = extract_horizon_data_from(
            data["cont"],
            horizon,
            mask=horizon_mask,
        )  # bs, bl, H

        future_action_from_cur = future_action_from_cur.reshape(
            -1, *future_action_from_cur.shape[2:]
        )  # bs * bl, H, *d
        start = {
            k: v.reshape(-1, *(v.shape[2:]))
            for k, v in post.items()
            if k not in NON_ROLLOUT_INPUTS and k != "atten"
        }  # bs * bl, *d

        time_embed_start = obs_input["time_embed"].reshape(
            -1, *obs_input["time_embed"].shape[2:]
        )
        prior_imagine = self.rssm.imagine(
            cast(future_action_from_cur),  # bs * bl, H , 5
            cast(start),  # bs * bl, *d
            step=step,  # int
            cont_decoder=self.decoder_heads["cont"],
            training=training,
            time_embed_start=time_embed_start,
        )  # bs * bl, H,*d
        # reshape from bs*bl, H, *d to bs, bl, H, *
        prior_imagine = {
            k: v.reshape(bs, bl, horizon, *v.shape[2:])
            for k, v in prior_imagine.items()
        }
        rollout_pred_reward = self.decoder_heads["reward"](prior_imagine)  # bs * bl, H
        rollout_pred_cont = self.decoder_heads["cont"](prior_imagine)  # bs * bl, H

        # per step
        # repeat in report because we want test/report and eval/report
        # for h in range(horizon):  # 0 to horizon - 1
        #     # zero out all horizon other than h: bs, bl, horizon. only bs, bl, h is 1
        #     h_mask = jnp.zeros_like(horizon_mask).at[:, :, h].set(1) * horizon_mask
        #     reward_stats = jaxutils.balance_stats(
        #         rollout_pred_reward,
        #         rollout_data_reward,
        #         0.9,
        #         1.1,
        #         mask=h_mask,
        #         reward_values=(
        #             self.reward_values
        #             if self.config.reward_head.dist == "onehot"
        #             else None
        #         ),
        #     )
        #     report.update(
        #         {f"rollout_reward_1_{k}_h={h + 1}": v for k, v in reward_stats.items()}
        #     )

        #     cont_stats = jaxutils.balance_stats(
        #         rollout_pred_cont,
        #         rollout_data_cont,
        #         mask=horizon_mask * h_mask,
        #     )
        #     report.update(
        #         {f"rollout_cont_{k}_h={h + 1}": v for k, v in cont_stats.items()}
        #     )

        # # draw precision/recall curve for rollout_cont
        # def sigmoid(x):
        #     return 1 / (1 + jnp.exp(-x))

        # cont_probs = sigmoid(rollout_pred_cont.distribution.logits)  # type: ignore
        # report.update(
        #     {
        #         "rollout_cont_prob": cont_probs,
        #         "rollout_cont_label": rollout_data_cont,
        #         "rollout_cont_mask": horizon_mask,
        #     }
        # )

        # prior_imagine = self.rssm.imagine(
        #     data["action"][:N_FIRST_BS, :N_FIRST_BL],  # bs, bl, d*
        #     start,  # bs, d*
        #     step=step,
        #     cont_decoder=self.decoder_heads["cont"],
        # )  # bs, bl, d*

        # if "decoder" in self.decoder_heads:
        #     recon = self.decoder_heads["decoder"].__call__(context, training=training)
        #     openl = self.decoder_heads["decoder"].__call__(
        #         prior_imagine, training=training
        #     )

        openl_reward = self.decoder_heads["reward"].__call__(
            prior_imagine, training=training
        )
        openl_cont = prior_imagine["cont"]

        if "atten" in post:
            report.update(report_atten(post, N_FIRST_BS, N_FIRST_BL, data, "atten"))

        # if "drop_x" in obs_input:
        #     report.update(
        #         report_atten(
        #             post,
        #             N_FIRST_BS,
        #             N_FIRST_BL,
        #             data,
        #             "atten",
        #             obs_input["drop_x"],
        #             "drop_x_atten",
        #         )
        #     )
        #     report.update(
        #         report_atten(
        #             post,
        #             N_FIRST_BS,
        #             N_FIRST_BL,
        #             data,
        #             "atten",
        #             1 - obs_input["drop_x"],
        #             "not_drop_x_atten",
        #         )
        #     )

        if self.config.entity_track_and:
            assert "atten1" in post, "atten1 not in context"
            assert "atten2" in post, "atten2 not in context"

        # if "atten1" in post:
        #     report.update(report_atten(post, N_FIRST_BS, N_FIRST_BL, data, "atten1"))
        # if "atten2" in post:
        #     report.update(report_atten(post, N_FIRST_BS, N_FIRST_BL, data, "atten2"))

        # visualize image
        # TRUTH_KEY = "log_image"
        truth_key_image = "image" if "image" in data else "small_image"

        if self.config.run.use_table:
            report["table_atten"] = post["atten"][:N_FIRST_BS, :N_FIRST_BL].reshape(
                -1, *post["atten"].shape[2:]
            )  # nhead, ne, ne
            if "atten1" in post:
                report["table_atten1"] = post["atten1"][
                    :N_FIRST_BS, :N_FIRST_BL
                ].reshape(-1, *post["atten1"].shape[2:])
            if "atten2" in post:
                report["table_atten2"] = post["atten2"][
                    :N_FIRST_BS, :N_FIRST_BL
                ].reshape(-1, *post["atten2"].shape[2:])

            report["table_entity_ids"] = data["entity_ids"][
                :N_FIRST_BS, :N_FIRST_BL
            ].reshape(-1, *data["entity_ids"].shape[2:])  # bs*bl, ne
            report["table_entity_pos"] = data["entity_pos"][
                :N_FIRST_BS, :N_FIRST_BL
            ].reshape(-1, *data["entity_pos"].shape[2:])
            report["table_entity_id"] = data["entity_ids"][
                :N_FIRST_BS, :N_FIRST_BL
            ].reshape(-1, *data["entity_ids"].shape[2:])
            report["table_avatar_pos"] = data["avatar_pos"][
                :N_FIRST_BS, :N_FIRST_BL
            ].reshape(-1, *data["avatar_pos"].shape[2:])

            report["table_sent_ids"] = data["sent_ids"][
                :N_FIRST_BS, :N_FIRST_BL
            ].reshape(-1, *data["sent_ids"].shape[2:])  # bs*bl, ns

            report["table_openl_reward"] = openl_reward.mode().reshape(
                -1, *openl_reward.mode().shape[2:]
            )
            report["table_openl_cont"] = openl_cont.reshape(-1, *openl_cont.shape[2:])
            report["table_reward_pred"] = reward_pred.mode().reshape(
                -1, *reward_pred.mode().shape[2:]
            )

            reward_data = data["reward"][:N_FIRST_BS, :N_FIRST_BL]
            report["table_reward_data"] = reward_data.reshape(
                -1, *reward_data.shape[2:]
            )

            if "image_loss_mean" in report:
                report["table_image_loss"] = report["image_loss_mean"]
            # else:
            #     # zero
            #     report["table_image_loss"] = jnp.zeros_like(reward_data)

            report["table_is_first"] = data["is_first"][
                :N_FIRST_BS, :N_FIRST_BL
            ].reshape(-1, *data["is_first"].shape[2:])

            cont_data = data["cont"][:N_FIRST_BS, :N_FIRST_BL]
            report["table_cont_data"] = cont_data.reshape(-1, *cont_data.shape[2:])
            report["table_cont_pred"] = cont_pred.mode().reshape(
                -1, *cont_pred.mode().shape[2:]
            )
            # report["table_step"] = data["time_step"][:N_FIRST_BS, :N_FIRST_BL].reshape(
            #     -1, *data["time_step"].shape[2:]
            # )
            report["table_action"] = data["action"][:N_FIRST_BS, :N_FIRST_BL].reshape(
                -1, *data["action"].shape[2:]
            )

            # for key in self.heads["decoder"].cnn_shapes.keys():
            # key = "image"
            if truth_key_image in data:
                assert self.config.has_image_input, "No image input"

                image_truth = data[truth_key_image][:N_FIRST_BS, :N_FIRST_BL].astype(
                    jnp.float32
                )
                if self.config.small_image:
                    multihot_truth = symbolic_to_multihot(image_truth)
                else:
                    multihot_truth = image_truth

                report["table_image_gt"] = image_truth.reshape(
                    -1, *image_truth.shape[2:]
                )
                assert multihot_truth.shape[-1] == 17, multihot_truth.shape
                report["table_image_gt_multihot"] = multihot_truth.reshape(
                    -1, *multihot_truth.shape[2:]
                )

            # if "decoder" in self.decoder_heads:
            #     for key in self.decoder_heads["decoder"].cnn_shapes.keys():
            #         if self.config.vis.recont:
            #             report["table_image_pred"] = (
            #                 recon[key]
            #                 .mode()[:, :N_FIRST_BL]
            #                 .reshape(-1, *recon[key].mode().shape[2:])
            #             )
            #             report["table_image_openl"] = (
            #                 openl[key].mode().reshape(-1, *openl[key].mode().shape[2:])
            #             )

        return report

    def _metrics(
        self,
        data,
        dists,
        post,
        prior,
        losses_not_scaled,
        priority_loss_per_bs,
        sup_loss_per_bs,
        rollout_loss_per_bs,
        step=None,
    ):
        """
        data: data from env
        dists: output of model
        """

        def entropy(feat):
            return get_dist_z_from(feat, self.config.rssm.impl, step).entropy()

        metrics = OrderedDict()
        metrics.update(jaxutils.tensorstats(entropy(prior), "prior_ent"))
        metrics.update(jaxutils.tensorstats(entropy(post), "post_ent"))
        metrics.update({"prior_dist": prior["logit"].reshape(-1)})
        metrics.update({"post_dist": post["logit"].reshape(-1)})

        for t in range(self.config.imag_horizon):
            t_mask = jnp.zeros_like(losses_not_scaled["dyn"]).at[:, t].set(1)
            metrics.update(
                {
                    f"dyn_loss_mean_t={t + 1}": (
                        losses_not_scaled["dyn"] * t_mask
                    ).sum()
                    / t_mask.sum()
                }
            )

        # find how many dyn_loss and rep_los >= 1 in batch
        metrics.update(
            {
                "dyn_loss_bigger_freebit_rate": (
                    losses_not_scaled["dyn"] > self.config.rssm_loss.dyn_free
                ).sum()
                / losses_not_scaled["dyn"].size
            }
        )
        if self.config.reward_head.dist == "onehot":
            tgt_reward = onehot_to_float(data["reward"], self.reward_values)
        else:
            tgt_reward = data["reward"]
        # zero
        metrics.update(
            {
                "dyn_reward_0_loss": (
                    losses_not_scaled["dyn"] * (tgt_reward == 0).astype(jnp.float32)
                ).sum()
                / (tgt_reward == 0).sum()
            }
        )

        # measure dyn loss where reward is 0.5 - 1
        metrics.update(
            {
                "dyn_reward_0.5_loss": (
                    losses_not_scaled["dyn"] * (tgt_reward == 0.5).astype(jnp.float32)
                ).sum()
                / (tgt_reward == 0.5).sum()
            }
        )
        metrics.update(
            {
                "dyn_reward_1_neg_loss": (
                    losses_not_scaled["dyn"] * (tgt_reward < -0.5).astype(jnp.float32)
                ).sum()
                / (tgt_reward < -0.5).sum()
            }
        )
        metrics.update(
            {
                "dyn_reward_1_pos_loss": (
                    losses_not_scaled["dyn"] * (tgt_reward > 0.5).astype(jnp.float32)
                ).sum()
                / (tgt_reward > 0.5).sum()
            }
        )
        for k, v in losses_not_scaled.items():
            if "rollout" not in k:
                metrics.update({f"{k}_loss_mean": v.mean()})
                metrics.update({f"{k}_loss_dist": v.reshape(-1)})
                metrics.update({f"{k}_loss_std": v.std()})

            else:
                horizon_mask_from_next = create_horizon_mask_from(
                    data["is_first"], data["is_last"], self.config.imag_horizon
                )

                metrics.update(
                    {
                        f"{k}_loss_mean": (v * horizon_mask_from_next).sum()
                        / horizon_mask_from_next.sum()
                    }
                )
                metrics.update({f"{k}_loss_dist": v.reshape(-1)})
                metrics.update({f"{k}_loss_std": v.std()})

                # # see loss for each horizon
                # for h in range(self.config.imag_horizon):
                #     h_mask = (
                #         jnp.zeros_like(horizon_mask_from_next).at[:, :, h].set(1)
                #         * horizon_mask_from_next
                #     )
                #     metrics.update(
                #         {
                #             f"{k}_loss_scaled_h={h + 1}": (v * h_mask).sum()
                #             / h_mask.sum()
                #         }
                #     )
                #     metrics.update({f"{k}_count_h={h + 1}": h_mask.sum()})

                # for t in range(self.config.imag_horizon):
                #     t_mask = (
                #         jnp.zeros_like(horizon_mask_from_next).at[:, t, :].set(1)
                #         * horizon_mask_from_next
                #     )
                #     metrics.update(
                #         {
                #             f"{k}_loss_scaled_t={t + 1}": (v * t_mask).sum()
                #             / t_mask.sum()
                #         }
                #     )
                #     metrics.update({f"{k}_count_t={t + 1}": t_mask.sum()})

                # rollout dyn
                if "rollout_dyn" == k:
                    reward_from_next = extract_horizon_data_from(
                        data["reward"],
                        self.config.imag_horizon,
                        horizon_mask_from_next,
                    )
                    if self.config.reward_head.dist == "onehot":
                        reward_from_next = onehot_to_float(
                            reward_from_next, self.reward_values
                        )
                    # rollout dyn where reward is 0.5 - 1
                    metrics.update(
                        {
                            "rollout_dyn_reward_1_pos": (
                                v * (reward_from_next > 0.5).astype(jnp.float32)
                            ).sum()
                            / (reward_from_next > 0.5).sum()
                        }
                    )
                    # rollout dyn where reward is -0.5 - -1
                    metrics.update(
                        {
                            "rollout_dyn_reward_1_neg": (
                                v * (reward_from_next < -0.5).astype(jnp.float32)
                            ).sum()
                            / (reward_from_next < -0.5).sum()
                        }
                    )
                    # # for each horizon h
                    # for h in range(self.config.imag_horizon):
                    #     h_mask = (
                    #         jnp.zeros_like(horizon_mask_from_next).at[:, :, h].set(1)
                    #         * horizon_mask_from_next
                    #     )
                    #     metrics.update(
                    #         {
                    #             f"rollout_dyn_reward_1_pos_h={h + 1}": (
                    #                 v
                    #                 * (reward_from_next > 0.5).astype(jnp.float32)
                    #                 * h_mask
                    #             ).sum()
                    #             / ((reward_from_next > 0.5) * h_mask).sum()
                    #         }
                    #     )
                    #     metrics.update(
                    #         {
                    #             f"rollout_dyn_reward_1_neg_h={h + 1}": (
                    #                 v
                    #                 * (reward_from_next < -0.5).astype(jnp.float32)
                    #                 * h_mask
                    #             ).sum()
                    #             / ((reward_from_next < -0.5) * h_mask).sum()
                    #         }
                    #     )

        if hasattr(dists["reward"], "kl"):
            metrics.update(
                {"reward_kl_dist": dists["reward"].kl(data["reward"]).reshape(-1)}
            )

        # temp
        # if step is not None:
        if self.config.run.opt_step:
            if self.config.rssm.soft_z.use:
                metrics["temperature"] = self.rssm.get_temp(step)

            # if self.config.actor.unimix_decay != "none":
            # get the first because value are duplicated across n_devices
            if self.config.use_unimix_decay:
                metrics["unimix_decay"] = get_unimix_decay(
                    self.config.actor.unimix_decay.init,
                    self.config.actor.unimix_decay.final,
                    self.config.actor.unimix_decay.steps,
                    step,
                )[0]

            if self.config.kl_anneal:
                metrics["kl_anneal"] = get_linear_increase_kl(
                    self.config.kl_anneal_param.init,
                    self.config.kl_anneal_param.final,
                    self.config.kl_anneal_param.steps,
                    step,
                    step_init=self.config.kl_anneal_param.step_init,
                )[0]

            if self.config.rep_anneal and step is not None:
                if self.config.rep_anneal_params.decay == "linear":
                    metrics["rep_anneal"] = get_linear_increase_kl(
                        self.config.rep_anneal_params.init,
                        self.config.rep_anneal_params.final,
                        self.config.rep_anneal_params.steps,
                        step,
                        step_init=self.config.rep_anneal_params.step_init,
                    )[0]
                elif self.config.rep_anneal_params.decay == "cubic":
                    metrics["rep_anneal"] = get_cubic_ease_in(
                        self.config.rep_anneal_params.init,
                        self.config.rep_anneal_params.final,
                        self.config.rep_anneal_params.steps,
                        step,
                        step_init=self.config.rep_anneal_params.step_init,
                    )[0]
                else:
                    raise ValueError("Invalid decay type")

            assert step is not None, "step is None"
            if not isinstance(step, int):
                if len(step.shape) == 1:
                    step = step[0]
                elif len(step.shape) >= 2:
                    raise ValueError(f"step shape is {step.shape}")
            metrics["opt_step"] = step.astype(jnp.int32)

        data_not_first = 1 - data["is_first"]
        # image
        if "image" in dists or "small_image" in dists:
            image_tag = "image" if "image" in dists else "small_image"
            metrics["image_pred_dist"] = dists[image_tag].mode().reshape(-1)
            image_tag_data = "image" if "image" in data else "small_image"
            if image_tag_data == "small_image":
                metrics["image_data_dist"] = symbolic_to_multihot(
                    data[image_tag_data]
                ).reshape(-1)
            else:
                metrics["image_data_dist"] = data[image_tag_data].reshape(-1)

            if dists[image_tag].distribution.logits is not None:
                metrics["image_logits_dist"] = dists[
                    image_tag
                ].distribution.logits.reshape(-1)
                metrics["image_probs_dist"] = jax.nn.sigmoid(
                    dists[image_tag].distribution.logits
                ).reshape(-1)
            else:
                assert (
                    self.config.decoder.cnn_sigmoid
                    or self.config.decoder_sent.cnn_sigmoid
                )
                metrics["image_probs_dist"] = dists[
                    image_tag
                ].distribution.probs.reshape(-1)
            metrics["image_pred_pos_rate"] = dists[image_tag].mode().mean()

            metrics["image_loss_is_first"] = (
                losses_not_scaled[image_tag] * data["is_first"]
            ).sum() / data["is_first"].sum()
            metrics["image_loss_not_first"] = (
                losses_not_scaled[image_tag] * data_not_first
            ).sum() / data_not_first.sum()
            metrics["image_loss_is_first_frac"] = (
                losses_not_scaled[image_tag] * data["is_first"]
            ).sum() / losses_not_scaled[image_tag].sum()

            # if self.config.use_read:
            # metrics["image_loss_mean_read"] = (
            #     losses[image_tag] * data["is_read_step"]
            # ).sum() / data["is_read_step"].sum()
            # metrics["image_loss_mean_read_frac"] = (
            #     losses[image_tag] * data["is_read_step"]
            # ).sum() / losses[image_tag].sum()

        # model
        # metrics["model_loss_mean"] = model_loss.mean()
        # metrics["model_loss_std"] = model_loss.std()
        metrics["priority_loss_per_batch"] = priority_loss_per_bs
        metrics["sup_loss_per_batch"] = sup_loss_per_bs
        metrics["rollout_loss_per_batch"] = rollout_loss_per_bs
        if "sample_id" in data:
            metrics["sample_id"] = data["sample_id"]

        if hasattr(dists["reward"], "kl"):
            reward_kl = dists["reward"].kl(data["reward"])
            metrics["reward_kl_mean"] = reward_kl.mean()
            metrics["reward_kl_is_first"] = (reward_kl * data["is_first"]).sum() / data[
                "is_first"
            ].sum()
            metrics["reward_kl_is_first_frac"] = (
                reward_kl * data["is_first"]
            ).sum() / reward_kl.sum()

            metrics["reward_data_dist"] = data["reward"].reshape(-1)
            metrics["reward_pred_dist"] = dists["reward"].mode().reshape(-1)

        # cont
        metrics["cont_data_dist"] = data["cont"].reshape(-1)
        sigmoid = lambda x: 1 / (1 + jnp.exp(-x))
        metrics["cont_pred_prob_dist"] = sigmoid(
            dists["cont"].distribution.logits
        ).reshape(-1)

        # rep and dyn
        # metrics["dyn_loss_dist"] = losses["dyn"].reshape(-1)
        metrics["dyn_loss_is_first_frac"] = (
            losses_not_scaled["dyn"] * data["is_first"]
        ).sum() / losses_not_scaled["dyn"].sum()
        metrics["dyn_loss_is_first"] = (
            losses_not_scaled["dyn"] * data["is_first"]
        ).sum() / data["is_first"].sum()
        metrics["dyn_loss_not_first"] = (
            losses_not_scaled["dyn"] * data_not_first
        ).sum() / data_not_first.sum()

        # metrics["rep_loss_dist"] = losses["rep"].reshape(-1)
        metrics["rep_loss_is_first_frac"] = (
            losses_not_scaled["rep"] * data["is_first"]
        ).sum() / losses_not_scaled["rep"].sum()
        metrics["rep_loss_is_first"] = (
            losses_not_scaled["rep"] * data["is_first"]
        ).sum() / data["is_first"].sum()
        metrics["rep_loss_not_first"] = (
            losses_not_scaled["rep"] * data_not_first
        ).sum() / data_not_first.sum()

        if "reward_indicator" in data:
            metrics.update(tensorstats(data["reward_indicator"], "reward_indicator"))

        if "game_id" in data:
            eps_game_id = jnp.where(
                data["is_first"],
                data["game_id"],
                -1,
            )
            metrics.update(tensorstats(eps_game_id, "game_id"))

        if "reward" in dists and not self.config.jax.debug_nans:
            # data_reward = (
            #     data["reward"]
            #     if self.config.reward_head.dist != "onehot"
            #     else float_to_onehot(data["reward"])
            # )
            data_reward = data["reward"]
            if "s1" not in self.config.task:
                stats = jaxutils.balance_stats(
                    dists["reward"],
                    data_reward,
                    0,
                    0.51,
                    # reward_onehot=self.config.reward_head.dist == "onehot",
                    reward_values=(
                        self.reward_values
                        if self.config.reward_head.dist == "onehot"
                        else None
                    ),
                )
                metrics.update({f"reward_0.5_{k}": v for k, v in stats.items()})

            stats = jaxutils.balance_stats(
                dists["reward"],
                data_reward,
                0.9,
                1.1,
                reward_values=(
                    self.reward_values
                    if self.config.reward_head.dist == "onehot"
                    else None
                ),
            )
            metrics.update({f"reward_1_{k}": v for k, v in stats.items()})
        # stats = jaxutils.tensorstats(dists["reward"].mode(), "reward_mode")

        # if "rollout_reward" in dists:
        #     # do multiple horizon
        #     for horizon in range(1, self.config.imag_horizon + 1):
        #         horizon_mask = create_horizon_mask_from(
        #             data["is_first"], data["is_last"], horizon
        #         )
        #         data_reward_horizon = extract_horizon_data_from(
        #             data["reward"], horizon, mask=horizon_mask
        #         )

        if "rollout_reward" in dists:
            horizon = self.config.imag_horizon
            horizon_mask_from_next = create_horizon_mask_from(
                data["is_first"], data["is_last"], horizon
            )  # bs, bl, H
            data_reward_horizon = extract_horizon_data_from(
                data["reward"],
                horizon,
                mask=horizon_mask_from_next,
                # [0, 0, 1, 0, 0] -> REWARD_VALUES[2] = 0
                mask_value_id=(
                    REWARD_VALUES[self.config.task].index(0)
                    if self.config.reward_head.dist == "onehot"
                    else None
                ),
            )  # bs, bl, H
            # is_first_horizon_mask: data['is_first']: bs, bl -> bs, bl, H
            is_first_horizon = data["is_first"][..., None]
            if self.config.overfit_batch:
                if hasattr(dists["rollout_reward"], "kl"):
                    metrics.update(
                        {
                            "rollout_reward_loss_full": (
                                dists["rollout_reward"].kl(data_reward_horizon)
                                * horizon_mask_from_next
                            )
                        }
                    )
                else:
                    assert self.config.reward_head.dist == "onehot", (
                        self.config.reward_head.dist
                    )
                    metrics.update(
                        {
                            "rollout_reward_loss_full": (
                                -dists["rollout_reward"].log_prob(data_reward_horizon)
                                * horizon_mask_from_next
                            )
                        }
                    )

            # rollout dyn loss full
            if self.config.overfit_batch:
                if "rollout_dyn" in dists:
                    metrics.update(
                        {"rollout_dyn_loss_full": losses_not_scaled["rollout_dyn_kl"]}
                    )

            # take the first bl
            stats = jaxutils.balance_stats(
                dists["rollout_reward"],
                data_reward_horizon,
                0,
                0.5,
                is_first_horizon * horizon_mask_from_next,
                reward_values=(
                    self.reward_values
                    if self.config.reward_head.dist == "onehot"
                    else None
                ),
                # take_is_first_only=True,
            )
            metrics.update(
                {f"rollout_is_first_reward_0.5_{k}": v for k, v in stats.items()}
            )
            stats = jaxutils.balance_stats(
                dists["rollout_reward"],
                data_reward_horizon,
                0.9,
                1.1,
                is_first_horizon * horizon_mask_from_next,
                reward_values=(
                    self.reward_values
                    if self.config.reward_head.dist == "onehot"
                    else None
                ),
                # take_is_first_only=True,
            )
            metrics.update(
                {f"rollout_is_first_reward_1_{k}": v for k, v in stats.items()}
            )
            stats = jaxutils.balance_stats(
                dists["rollout_reward"],
                data_reward_horizon,
                0.9,
                1.1,
                horizon_mask_from_next,
                reward_values=(
                    self.reward_values
                    if self.config.reward_head.dist == "onehot"
                    else None
                ),
            )
            metrics.update({f"rollout_reward_1_{k}": v for k, v in stats.items()})
            # for h in range(horizon):
            #     h_mask = (
            #         jnp.zeros_like(horizon_mask_from_next).at[:, :, h].set(1)
            #         * horizon_mask_from_next
            #     )
            #     stats = jaxutils.balance_stats(
            #         dists["rollout_reward"],
            #         data_reward_horizon,
            #         0.9,
            #         1.1,
            #         h_mask,
            #         reward_values=(
            #             self.reward_values
            #             if self.config.reward_head.dist == "onehot"
            #             else None
            #         ),
            #     )
            #     metrics.update(
            #         {f"rollout_reward_1_{k}_h={h + 1}": v for k, v in stats.items()}
            #     )

            if self.reward_values is not None:
                data_reward_horizon = self.reward_values[
                    jnp.argmax(data_reward_horizon, axis=-1)
                ]  # bs, bl, H
                # jax.debug.print("real {res}", res=data_reward_horizon)

            if self.reward_values is not None:
                rollout_reward_mode = dists["rollout_reward"].mode(self.reward_values)
            else:
                rollout_reward_mode = dists["rollout_reward"].mode()

            # mask out with horizon_mask
            rollout_reward_mode = jnp.where(
                horizon_mask_from_next, rollout_reward_mode, MASK_VALUE_DIST
            )
            metrics["rollout_reward_pred_dist"] = rollout_reward_mode.reshape(-1)
            data_reward_horizon = jnp.where(
                horizon_mask_from_next, data_reward_horizon, MASK_VALUE_DIST
            )
            metrics["rollout_reward_data_dist"] = data_reward_horizon.reshape(-1)

        if "rollout_cont" in dists:
            horizon = self.config.imag_horizon
            horizon_mask_from_next = create_horizon_mask_from(
                data["is_first"], data["is_last"], horizon
            )  # bs, bl, H
            data_cont_horizon = extract_horizon_data_from(
                data["cont"], horizon, mask=horizon_mask_from_next
            )
            is_first_horizon = data["is_first"][..., None]  # bs, bl, 1
            # jax.debug.print(
            #     "target mask {res}",
            #     res=(data_horizon * is_first_horizon)[0],
            # )
            # stats = jaxutils.balance_stats(
            #     dists["rollout_cont"],
            #     data_cont_horizon,
            #     mask=horizon_mask * is_first_horizon,
            # )
            # metrics.update({f"rollout_cont_is_first_{k}": v for k, v in stats.items()})

            stats = jaxutils.balance_stats(
                dists["rollout_cont"],
                data_cont_horizon,
                mask=horizon_mask_from_next,
            )
            metrics.update({f"rollout_cont_{k}": v for k, v in stats.items()})
            metrics["rollout_cont_pred_dist"] = dists["rollout_cont"].mode().reshape(-1)
            metrics["rollout_cont_data_dist"] = data_cont_horizon.reshape(-1)
            metrics["rollout_cont_is_first_pred_dist"] = jnp.where(
                horizon_mask_from_next * is_first_horizon,
                dists["rollout_cont"].mode(),
                MASK_VALUE_DIST,
            ).reshape(-1)

            metrics["rollout_cont_is_first_data_dist"] = jnp.where(
                horizon_mask_from_next * is_first_horizon,
                data_cont_horizon,
                MASK_VALUE_DIST,
            ).reshape(-1)

        if "reward_indicator" in data:
            metrics["reward_indicator_dist"] = data["reward_indicator"].reshape(-1)
            metrics["reward_indicator_pos_rate"] = (
                data["reward_indicator"] > 0
            ).sum() / data["reward_indicator"].size

        if "cont" in dists and not self.config.jax.debug_nans:
            stats = jaxutils.balance_stats(dists["cont"], data["cont"], 0.5)
            metrics.update({f"cont_{k}": v for k, v in stats.items()})

        if "prev_action" in dists:
            action_pred = jnp.argmax(dists["prev_action"].logits, axis=-1)
            action_gt = jnp.argmax(data["prev_action"], axis=-1)
            not_first_step = 1 - data["is_first"]
            action_acc = (
                (action_pred == action_gt) * not_first_step
            ).sum() / not_first_step.sum()
            # mean over bs, bl
            metrics["action_acc"] = action_acc
            metrics["action_pred"] = action_pred.reshape(-1)

        return metrics


def report_atten(
    context, N_FIRST_BS, N_FIRST_BL, data, tag, mask=None, metrics_tag=None
):
    assert "atten" in tag, tag
    report = {}
    # context['atten']: bs, bl, nhead, ne, ne
    # sum over nhead -> bs, bl, 1, ne, ne
    # context[tag] = context[tag].sum(2, keepdims=True)  # bs, bl, 1, ne, ne

    num_heads = context[tag].shape[2]
    for head in range(num_heads):
        head_scores = context[tag][:, :, head, :, :]  # bs, bl, ne, ne
        entropy = head_scores * jnp.log(head_scores + 1e-9)
        entropy = -entropy.sum(-1)  # bs, bl, ne
        report[f"entropy_mean_{tag}_{head=}"] = entropy.mean()

        argmax_atten_sent = jnp.argmax(head_scores, -1)  # (bs, bl, ne)
        argmax_atten_sent = argmax_atten_sent[:N_FIRST_BS, :N_FIRST_BL]

        # (bs, bl, ne)
        argmin_atten_sent = jnp.argmin(head_scores, -1)
        argmin_atten_sent = argmin_atten_sent[:N_FIRST_BS, :N_FIRST_BL]

        # data['manual_ids']: (bs, bl, ns) - gt sentence for each entity
        nonzero_entity_ids = data["entity_ids"][:N_FIRST_BS, :N_FIRST_BL, :] > 0
        if mask is not None:
            nonzero_entity_ids = nonzero_entity_ids * mask[:N_FIRST_BS, :N_FIRST_BL, :]

        if metrics_tag is None:
            metrics_tag = tag

        report[f"acc_{metrics_tag}_{head=}"] = (
            (argmax_atten_sent == data["manual_ids"][:N_FIRST_BS, :N_FIRST_BL, :])
            * nonzero_entity_ids
        ).sum() / nonzero_entity_ids.sum()

        report[f"acc_{metrics_tag}_min_{head=}"] = (
            (argmin_atten_sent == data["manual_ids"][:N_FIRST_BS, :N_FIRST_BL, :])
            * nonzero_entity_ids
        ).sum() / nonzero_entity_ids.sum()

    return report


# %%
# REWARD_VALUES = {
#     "s1": [-1, 0, 1],
#     "messenger_s1": [-1, 0, 1],
#     "s2": [-1, 0, 0.5, 1],
#     "messenger_s2": [-1, 0, 0.5, 1],
#     "s3": [-2, -1, 0, 0.5, 1],
#     "messenger_s3": [-2, -1, 0, 0.5, 1],
#     "easy": [-1, 0, 0.5, 1],
#     "lwm_easy": [-1, 0, 0.5, 1],
#     "medium": [-1, 0, 0.5, 1],
#     "lwm_medium": [-1, 0, 0.5, 1],
#     "hard": [-1, 0, 0.5, 1],
#     "lwm_hard": [-1, 0, 0.5, 1],
# }


def test():
    print(REWARD_VALUES["s1"].index(0))


if __name__ == "__main__":
    test()
