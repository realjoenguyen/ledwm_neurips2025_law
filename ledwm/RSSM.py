from einops import repeat
from termcolor import cprint

# from MLP import MLP
from ledwm import jaxutils, ninjax as nj
import ledwm.nets.Dist
from ledwm.nets.EncoderRSSM import EncoderRSSM
from ledwm.nets.GRUCell import GRUCell
from ledwm.nets.Initializer import Initializer
from ledwm.nets.Linear import LinearAct
from ledwm.constants import NON_ROLLOUT_INPUTS
from ledwm.embodied.envs.MessengerSent import (
    NUM_ENTITIES,
    NUM_SENTS,
)
from ledwm.nets.gumbel_softmax import GumbelSoftmax
from ledwm.nets import f32
import ledwm.tfp_compat  # noqa: F401
from tensorflow_probability.substrates.jax import distributions as tfd
from ledwm.nets.residual import ResidualMLP
from .jaxutils import add_dummy_first_action, apply_dropout_on, cast_to_compute as cast
from .jaxutils import sg, tree_map, get_dist_z_from, sample_z_from
import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
from ledwm.nets.EncoderRSSM import ZGenerator


# bs, bl, d* -> bl, bs, d*
def swap_bs_bl(x):
    return x.transpose([1, 0] + list(range(2, len(x.shape))))


ENCODER_RSSM_KEY = "encoder_rssm"


class RSSM(nj.Module):
    def __init__(
        self,
        impl="softmax",
        deter=1024,
        stoch=32,
        classes=32,
        unroll=False,
        unimix=0,
        action_clip=1.0,
        # bottleneck=-1,  # -1
        maskgit={},
        symlog_inputs=False,
        cnn_atten={},
        cnn_image={},
        atten=None,
        atten_concepts=None,
        add_h_to_query=False,
        mlp=None,
        image_shape=[16, 16],
        use_atten=True,
        env_cache=None,
        read=False,
        task="none",
        soft_z={},
        deter_first="zero",
        concat_deter=True,
        dropout=0,
        concat_image=True,
        image_depths=None,
        image_kernels=None,
        has_image_input=True,
        seperate_cnn_image=False,
        entity_track_z=False,
        entity_track_atten=False,
        pos_dim=32,
        dead_vec=False,
        residual=None,
        config=None,
        num_gru_layers=1,
        gru_multi_layers=False,
        lang_image_recon=False,
        no_history_rollout=False,
        **kw,
    ):
        assert task != "none", task
        assert config is not None
        self.gru_multi_layers = gru_multi_layers
        self.num_gru_layers = num_gru_layers
        self.config = config
        self.read = read
        self.pos_dim = pos_dim
        self.dead_vec = dead_vec
        self.entity_track_z = entity_track_z
        self.entity_track_atten = entity_track_atten
        self.image_kernels = image_kernels
        self.seperate_cnn_image = seperate_cnn_image
        self.concat_image = concat_image
        self.soft_z = soft_z
        self.deter_first = deter_first
        self.concat_deter = concat_deter
        if self.soft_z["use"]:
            cprint("rssm.config | distribution=gumbel_softmax", "red")
        self.task = task
        assert impl in ("gaussian", "softmax", "maskgit"), impl
        self._impl = impl
        self.deter = deter
        self.stoch = stoch
        self.classes = classes
        self._unroll = unroll
        self._unimix = unimix
        self._action_clip = action_clip
        # self._bottleneck = bottleneck
        self._image_shape = image_shape
        self.kw = kw
        self.residual_kw = residual
        self.env_cache = env_cache
        self.use_opt_step = soft_z["use"]
        self._num_gru_layers = num_gru_layers
        self.lang_image_recon = lang_image_recon
        self.no_history_rollout = no_history_rollout
        if self._impl == "maskgit":
            raise NotImplementedError
        # self._kw_atten = {"heads": head_atten, "size": d_atten}

        self.use_atten = use_atten
        self.prior_z_gen = ZGenerator(
            self._impl,
            self.stoch * NUM_ENTITIES[self.task]
            if self.config.seperate_object_z
            else self.stoch,
            self.classes,
            self._unimix,
            self.config,
            **self.kw,
            name="prior_z_gen",
        )
        self.encoder_rssm = EncoderRSSM(
            entity_track_atten,
            config,
            cnn_atten,
            cnn_image,
            **kw,
            name=ENCODER_RSSM_KEY,
        )

        if self.config.slow_encoder_fraction > 0:
            self.encoder_rssm_slow = EncoderRSSM(
                entity_track_atten,
                config,
                cnn_atten,
                cnn_image,
                **kw,
                name=f"{ENCODER_RSSM_KEY}_slow",
            )
            self.encoder_updater = jaxutils.SlowUpdater(
                self.encoder_rssm,
                self.encoder_rssm_slow,
                self.config.slow_encoder_fraction,
                self.config.slow_encoder_update,
            )

        self.dropout = dropout

        # s2 only
        self.dropout_pos_x = nj.FlaxModule(nn.Dropout, rate=0.5, name="dropout_pos_x")

        self.add_h_to_query = add_h_to_query
        if add_h_to_query:
            cprint("rssm.config | add_hidden_to_query=true", "green")
        else:
            cprint("rssm.config | add_hidden_to_query=false", "yellow")

    def initial(
        self,
        batch_size,
        verbose=False,
        use_for_policy=False,
    ):
        state = dict(
            deter=jnp.zeros([batch_size, self.deter], f32),
            logit=jnp.zeros([batch_size, self.stoch, self.classes], f32),
            stoch=jnp.zeros([batch_size, self.stoch, self.classes], f32),
        )

        if self.config.seperate_object_z:
            state["logit"] = jnp.zeros(
                [batch_size, self.stoch * NUM_ENTITIES[self.task], self.classes],
                f32,
            )
            state["stoch"] = jnp.zeros(
                [batch_size, self.stoch * NUM_ENTITIES[self.task], self.classes],
                f32,
            )

        # self.init_deter = self.get(
        #     "init_deter",
        #     Initializer(),
        #     [self.deter],
        # )
        # # repeat to batch_size, using einsum
        # state["deter"] = repeat(self.init_deter, "d -> b d", b=batch_size).astype(f32)

        if self.config.action_pred and not use_for_policy:
            state["x"] = jnp.zeros([batch_size, self.kw["units"]], f32)
            if "prev_deter" in self.config.action_head.inputs:
                state["prev_deter"] = jnp.zeros([batch_size, self.deter], f32)

            if "prev_stoch" in self.config.action_head.inputs:
                state["prev_stoch"] = jnp.zeros(
                    [batch_size, self.stoch, self.classes], f32
                )  # doesn't matter anyway bc we mask it (is_first=True)

                if self.config.seperate_object_z:
                    state["prev_stoch"] = jnp.zeros(
                        [
                            batch_size,
                            self.stoch * NUM_ENTITIES[self.task],
                            self.classes,
                        ],
                        f32,
                    )

        if verbose and not self.config.movement_role_lang_grid:
            state["atten"] = jnp.zeros(
                [
                    batch_size,
                    self.config.rssm.atten.heads,
                    NUM_ENTITIES[self.task],
                    NUM_SENTS[self.task],
                ],
                f32,
            )

        if self.gru_multi_layers:
            assert self.num_gru_layers > 1, self.num_gru_layers
            del state["deter"]
            # state["deter_layers"] = jnp.zeros(
            #     [batch_size, self.num_gru_layers, self.deter], f32
            # )
            self.init_deter_layers = self.get(
                "init_deter_layers",
                Initializer(),
                [self.num_gru_layers, self.deter],
            )
            state["deter_layers"] = repeat(
                self.init_deter_layers, "l d -> b l d", b=batch_size
            ).astype(f32)

        if self.lang_image_recon and not use_for_policy:
            state["lang_image"] = jnp.zeros([batch_size, 12, 12, 64], f32)

        return cast(state)

    def observe(
        self,
        obs_input,  # bs, bl
        action,  # bs, bl
        is_first,  # bs, bl
        state=None,  # latent
        step=None,
        encoder_type="token",
        training=True,
        verbose=False,
    ):
        # action[t] is the action for the next obs so we need to shift action by 1
        # now: action[t] is the action lead to obs[t]
        # dummy first action
        action = add_dummy_first_action(action)

        if encoder_type == "token":
            raise NotImplementedError

        elif encoder_type == "sent":
            # embed = obs_input
            assert isinstance(obs_input, dict)
            # 'entity_embed', 'entity_pos', 'avatar_embed', 'avatar_pos', 'image', 'sent_embed', 'read_embed'
            inputs = {
                "action": swap_bs_bl(action),
                "is_first": swap_bs_bl(is_first),
                **{k: swap_bs_bl(obs_input[k]) for k in obs_input.keys()},
            }

            def step_fn(state, inputs):
                return self.obs_step_sent(
                    step, state, inputs, training=training, verbose=verbose
                )
        else:
            raise NotImplementedError

        batch_size = action.shape[0]
        state = state or self.initial(batch_size, verbose)  # latent state, action
        state = self.recurrent_state(state)

        def scan_step_fn(state, inputs):
            post = step_fn(state, inputs)
            return self.recurrent_state(post), post

        # (bl, bs, d) - swap bs and bl
        post = jaxutils.scan_with_output(scan_step_fn, inputs, state, self._unroll)
        # (bs, bl, d) - swap bs and bl again
        post = {k: swap_bs_bl(v) for k, v in post.items()}
        # if self.config.drop_x_randomly_in_query and training:
        #     post["drop_x"] = drop_x
        return post

    def recurrent_state(self, state):
        return jaxutils.compact_rssm_recurrent_state(state, self.gru_multi_layers)

    def imagine(
        self,
        actions,  # bs, horizon, 5
        start,  # bs, d*
        step=None,  #
        cont_decoder=None,
        training=True,
        time_embed_start=None,
    ):
        """
        returns:  bs, horizon, d*
        """
        # from ledwm.WM import ACTION_PRED_KEYS

        start = {
            k: v
            for k, v in start.items()
            if k not in NON_ROLLOUT_INPUTS and k not in ["lang_image", "decoder"]
        }

        start = start or self.initial(actions.shape[0], use_for_policy=True)  # bs, d*

        # swap because to use jax.scan, we need to have the first dimension as the time dimension
        actions = swap_bs_bl(actions)  # (horizon, bs*bl, num_act)
        # first_cont = (1.0 - state["is_terminal"]).astype(jnp.float32)

        # put dropout to start_state['deter']
        # if self.dropout > 0:
        #     start_state["deter"] = apply_dropout_on(
        #         start_state["deter"], self.dropout_func, training, step
        #     )
        if self.no_history_rollout:
            if "deter" in start:
                # linear of time_step_start
                if self.config.time_in == "deter":
                    start["deter"] = self.get(
                        "time_step_start_linear",
                        LinearAct,
                        act="silu",
                        norm="layer",
                        units=start["deter"].shape[-1],
                    )(time_embed_start)  # bs * bl, d_deter
                    start["deter"] = cast(start["deter"])

                else:
                    start["deter"] = jaxutils.apply_zero_deter(
                        start["deter"], training, step
                    )

                # bs = start_state["deter"].shape[0]
                # start_state["deter"] = repeat(self.init_deter, "d -> b d", b=bs).astype(
                #     jnp.float16
                # )
            else:
                assert "deter_layers" in start, start.keys()
                if self.config.time_in == "deter":
                    raise NotImplementedError
                else:
                    start["deter_layers"] = jaxutils.apply_zero_deter(
                        start["deter_layers"], training, step
                    )
                    start["deter_layers"] = cast(start["deter_layers"])

        carry = {}
        if self.config.imag_cont_hard:
            if cont_decoder is not None:
                carry["cont"] = cont_decoder(start, training=False).mode()

        def step_fn_imagine(cur, action):
            cur_state, carry = cur
            state = self.imagine_step(cur_state, action, step=step)
            if self.config.imag_cont_hard:
                if cont_decoder is not None:
                    carry["cont"] = (
                        cont_decoder(state, training=False).mode() * carry["cont"]
                    )  # if cont==0 then future is always 0
            return state, carry

        prior = jaxutils.scan(
            step_fn_imagine,
            actions,  # (horizon, bs*bl, num_act): inputs
            (start, carry),  # (bs*bl, *) : start
            self._unroll,
        )  # horizon, bs*bl, d*

        if isinstance(prior, tuple):
            # get all k,v from prior[0] and prior[1]
            prior = {k: v for p in prior for k, v in p.items()}

        if not self.config.imag_cont_hard and cont_decoder is not None:
            if self.config.imag_cont == "mode":
                cont = cont_decoder(prior).mode()
            elif self.config.imag_cont == "mean":
                cont = cont_decoder(prior).mean()
            else:
                raise NotImplementedError(self.config.imag_cont)

            # prior["cont"] = jnp.concatenate([first_cont[None], cont[1:]], 0)
            prior["cont"] = cont

        prior = {k: swap_bs_bl(v) for k, v in prior.items()}  # bs*bl, horizon,
        return prior

    def obs_step_sent(
        self,
        step,  # -1 if eval, test
        state,
        # action,  # action[t] that leads to obs[t]
        # is_first,
        # entity_embed,
        # entity_pos,
        # avatar_embed,
        # avatar_pos,
        # time_embed=None,  # bs, bl, d
        # time_step=None,  # bs
        # sent_embed=None,
        # dp=None,
        # movement_embed=None,  # bs, Ne, d
        # image=None,
        # gt_grounding_scores=None,  # bs, bl, Ne, Ns
        # info_embed=None,
        # drop_x=None,
        data,
        training=True,  # if train = True -> train + might be eval test -> use step to determine
        verbose=False,
        use_for_policy=False,
    ):
        """
        state has
        deter: history h_t-1 all history until obs[t-2]
        stoch: z_t-1: obs t-1 representation
        action: action_t-1: action that leads to obs[t]

        """
        # if time_embed is not None:
        #     assert time_embed.shape[-1] == 32, time_embed.shape

        # h_t (bs, d_deter)
        if "atten" in state:
            state = {k: v for k, v in state.items() if "atten" not in k}

        # history over prev_state and prev_action
        gru_output = self._gru(
            state,
            data["action"],
            data["is_first"],
            use_for_policy=use_for_policy,
            training=training,
        )

        stats, stoch, encoded_obs, info = self.encoder_rssm.__call__(
            data,
            gru_output["deter"],
            step,
            training,
            verbose,
        )

        if self.gru_multi_layers:
            gru_output.pop("deter")

        post = {"stoch": stoch, **gru_output, **stats}

        if self.config.action_pred:
            post["x"] = encoded_obs
            post["prev_stoch"] = state["stoch"]
            if "prev_deter" in state:
                assert "prev_deter" in self.config.action_head.inputs, (
                    self.config.action_head.inputs
                )
                if self.gru_multi_layers:
                    post["prev_deter"] = state["deter_layers"][..., -1, :]
                else:
                    post["prev_deter"] = state["deter"]

        if verbose:
            post.update(info)

        if self.lang_image_recon:
            raise NotImplementedError
            # W, H, D = info["attn_grid"].shape[-3:]
            # batch_dims = info["attn_grid"].shape[:-3]
            # # bs, 12, 12, 64
            # post["lang_image"] = jnp.zeros((*batch_dims, 12, 12, 64), f32)
            # post["lang_image"] = post["lang_image"].at[..., :W, :H, :].set(attn_grid)

        cprint("rssm.posterior", "yellow")
        total_param_post = sum([np.prod(v.shape) for k, v in post.items()])
        for k, v in post.items():
            # print # of parameters and their % of total
            cprint(
                f"rssm.state_field | state=posterior | name={k} | "
                f"shape={v.shape} | fraction={np.prod(v.shape) / total_param_post:.2%}"
            )

        return post

    def imagine_step(
        self,
        cur_state,
        cur_action,
        cur_is_first=None,
        step=None,
        training=True,
    ):
        # imagine h and z
        gru_output = self._gru(cur_state, cur_action, cur_is_first)
        state = self._prior(
            gru_output["deter"],
            sample=True,
            step=step,
            argmax=self.config.z_argmax_imagine,
        )
        if "deter_layers" in gru_output:
            state["deter_layers"] = gru_output["deter_layers"]
            del state["deter"]

        cprint("rssm.imagine_state", "yellow")
        total = sum([np.prod(v.shape) for k, v in state.items()])
        for k, v in state.items():
            cprint(
                f"rssm.state_field | state=imagine | name={k} | shape={v.shape} | "
                f"fraction={np.prod(v.shape) / total:.2%}"
            )

        return cast(state)

    def get_temp(self, step):
        assert step is not None
        interval = self.soft_z["interval"]
        init = self.soft_z["init"]
        init = jnp.array(init, f32)
        anneal_rate = self.soft_z["annel_rate"]
        # if step.shape = (1,) then step = step[0]

        if not isinstance(step, int):
            if len(step.shape) == 1:
                step = step[0]
            else:
                step = step[0][0]

        should_update = step % interval == 1
        temp_prev = init * jax.lax.exp(-anneal_rate * jnp.maximum(0, step - interval))

        temp = jax.lax.cond(
            should_update,
            lambda _: init * jax.lax.exp(-anneal_rate * step),
            lambda _: temp_prev,
            operand=None,
        )
        return jnp.maximum(self.soft_z["final"], temp)

    # @staticmethod
    # def get_dist(self, stats, step=None):

    def loss(self, post, step=None, dyn_free=1.0, rep_free=1.0, data=None):
        if self.gru_multi_layers:
            deter = post["deter_layers"][..., -1, :]
        else:
            deter = post["deter"]
        prior = self._prior(deter, sample=False, step=step)

        dyn = None
        if self._impl == "gaussian":
            raise NotImplementedError

        if self._impl == "softmax":
            # prediction loss: prior is learnt towards sg(post)
            # post: get logits learnt from obs
            z_sg = get_dist_z_from(sg(post), self._impl, step)
            z_hat = get_dist_z_from(prior, self._impl, step)
            dyn = z_sg.kl_divergence(z_hat)
            # regulize the actual representation z_t to match with the prior z^hat_t - predictable from history h_t
            z = get_dist_z_from(post, self._impl, step)
            z_hat_sg = get_dist_z_from(sg(prior), self._impl, step)
            rep = z.kl_divergence(z_hat_sg)

        if self._impl == "maskgit":
            raise NotImplementedError

        assert dyn is not None and rep is not None, "Invalid impl"
        real_dyn = dyn
        if dyn_free > 0:
            dyn = jnp.maximum(dyn, dyn_free)
        if rep_free > 0:
            rep = jnp.maximum(rep, rep_free)  # bs, bl

        if self.config.kl_anneal:
            # assert dyn_free == 0.0 and rep_free == 0.0
            kl_weight = ledwm.nets.Dist.get_linear_increase_kl(
                self.config.kl_anneal_params.init,
                self.config.kl_anneal_params.final,
                self.config.kl_anneal_params.steps,
                step,
                step_init=self.config.kl_anneal_params.step_init,
            )
            dyn = dyn * kl_weight
            rep = rep * kl_weight

        if self.config.rep_anneal:
            # assert dyn_free == 0.0 and rep_free == 0.0
            assert step is not None, "step is None"
            rep_weight = None
            if self.config.rep_anneal_params.decay == "linear":
                rep_weight = ledwm.nets.Dist.get_linear_increase_kl(
                    self.config.rep_anneal_params.init,
                    self.config.rep_anneal_params.final,
                    self.config.rep_anneal_params.steps,
                    step,
                    step_init=self.config.rep_anneal_params.step_init,
                )  # (bs, )

            elif self.config.rep_anneal_params.decay == "cubic":
                rep_weight = ledwm.nets.Dist.get_cubic_ease_in(
                    self.config.rep_anneal_params.init,
                    self.config.rep_anneal_params.final,
                    self.config.rep_anneal_params.steps,
                    step,
                    step_init=self.config.rep_anneal_params.step_init,
                )  # (bs, )
            else:
                raise ValueError("Invalid decay type")
            assert rep_weight is not None, rep_weight
            rep = rep * rep_weight[:, None]
            if self.config.dyn_rep_anneal:
                dyn = dyn * rep_weight[:, None]

        if self.config.dyn_rep_up_weight > 0:
            # up weight in the first 3 steps
            FIRST_UP_STEPS = 3
            dyn = dyn.at[:, FIRST_UP_STEPS:].set(
                dyn[:, FIRST_UP_STEPS:] * self.config.dyn_rep_up_weight
            )
            rep = rep.at[:, FIRST_UP_STEPS:].set(
                rep[:, FIRST_UP_STEPS:] * self.config.dyn_rep_up_weight
            )
        if self.config.dyn_rep_up_nonzero > 0:
            # up weight in nonzero rewards
            assert data is not None and "reward" in data
            reward = data["reward"]
            dyn = jnp.where(reward != 0, dyn * self.config.dyn_rep_up_nonzero, dyn)
            rep = jnp.where(reward != 0, rep * self.config.dyn_rep_up_nonzero, rep)

        return {"dyn": dyn, "rep": rep, "real_dyn": real_dyn}, prior

    def _prior(self, deter, sample, step=None, argmax=False):
        # if sample=False: stoch = None
        # elif argmax=True: z = argmax(prior)
        # else: sample z from the prior deter

        # if self._impl == "gaussian":
        #     raise NotImplementedError

        # elif self._impl == "softmax":
        #     if self.config.residual_z:
        #         # deter=512, units=512
        #         x = self.get(
        #             "img_out",
        #             ResidualMLP,
        #             units=self.kw["units"],
        #             num_blocks=self.residual_kw["num_blocks"],
        #         )(deter)

        #     else:
        #         x = self.get("img_out", LinearAct, **self.kw)(deter)

        # stats = self._stats("img_stats", x)  # return logit
        stats = self.prior_z_gen.__call__(deter)
        z_dist = get_dist_z_from(stats, self._impl, step)
        stoch = sample_z_from(z_dist, argmax)
        return cast({"deter": deter, "stoch": stoch, **stats})

        # else:
        #     raise NotImplementedError

    def _gru_multi_layers(self, deter, x_input):
        # assert prev_deter has >= 3 dims
        # (bs, layer, d_deter) or (bs, bl, layer, d_deter)
        assert deter.ndim >= 3, deter.shape
        h_l = None
        for layer in range(self.num_gru_layers):
            prev_time_deter = deter[:, layer, :]
            h_l = self.get(
                f"gru_cell_{layer}",
                GRUCell,
                self.deter,
                self.kw["norm"],
                self.kw["winit"],
                layer,
            )(prev_time_deter, x_input)

            if layer > 0 or self.config.skip_first_layer:
                x_input = h_l["deter"] + x_input
            else:
                x_input = h_l["deter"]
            deter = deter.at[:, layer, :].set(h_l["deter"])

        assert h_l is not None
        return {"deter": h_l["deter"], "deter_layers": deter}

    def _gru(
        self,
        cur_state,
        cur_action,
        is_first=None,
        use_for_policy=False,
        training=False,
    ):
        # prev_state: (bs, d_deter)
        # prev_action: (bs, num_act)
        cur_action = cast(cur_action)

        if is_first is not None:
            batch_size = is_first.shape[0]
            initial_state = self.initial(
                batch_size, use_for_policy=use_for_policy
            )
            initial_state = {key: initial_state[key] for key in cur_state}
            cur_state, cur_action = tree_map(
                lambda prev, init: jaxutils.switch(is_first, init, prev),
                (cur_state, cur_action),
                (
                    initial_state,
                    jnp.zeros_like(cur_action),
                ),
            )
        batch_shape = cur_state["stoch"].shape[:-2]  # bs, bl
        x = jnp.concatenate(
            [
                cur_state["stoch"].reshape((*batch_shape, -1)),  # bs, bl, *
                cur_action.reshape((*batch_shape, -1)),
            ],
            -1,
        )

        if self.config.residual_z:
            x = self.get(
                "x_input",
                ResidualMLP,
                units=self.deter,
                num_blocks=self.residual_kw["num_blocks"],
            )(x)
        else:
            x = self.get(
                "x_input",
                LinearAct,
                self.deter,
                norm=self.kw["norm"],
                act=self.kw["act"],
            )(x)
            # x = apply_dropout_on(x, self._dropout, training=training)

        if self.gru_multi_layers:
            return self._gru_multi_layers(cur_state["deter_layers"], x)
        else:
            return self.get(
                "gru_cell",
                GRUCell,
                self.deter,
                self.kw["norm"],
                self.kw["winit"],
                0,
            )(cur_state["deter"], x)

    # def _stats(self, name, x, unimix=False) -> dict:
    #     # return 'logit'
    #     # can be prior or posterior
    #     # if prior: x = state['deter']
    #     # if posterior: x = state['stoch', 'deter']

    #     if self._impl == "gaussian":
    #         # raise NotImplementedError
    #         x = self.get(name, LinearAct, 2 * self.stoch)(x)
    #         print("Linear x:", x.shape)
    #         # bs, bl, 2 * d_stoch=32
    #         mean, std = jnp.split(x, 2, -1)
    #         std = jax.nn.softplus(std) + 0.1
    #         return {"mean": mean, "std": std}

    #     elif self._impl == "softmax":
    #         # bs, d_unit -> bs, d_stoch * n_class -> bs, d_stoch, n_class -> softmax -> mix with uniform -> logit: bs, d_stoch, n_class
    #         # linear only
    #         x = self.get(
    #             name,
    #             LinearAct,
    #             self.stoch * self.classes,
    #             norm=self.kw["norm"],
    #             act="silu",
    #         )(x)

    #         # bs, d_stoch, n_class
    #         logit = x.reshape(x.shape[:-1] + (self.stoch, self.classes))
    #         if unimix:
    #             if self._unimix:
    #                 probs = jax.nn.softmax(logit, -1)  # prob: bs, d_stoch, n_class
    #                 uniform = jnp.ones_like(probs) / probs.shape[-1]
    #                 # mix with uniform
    #                 probs = (1 - self._unimix) * probs + self._unimix * uniform
    #                 logit = jnp.log(probs)

    #         return {"logit": logit}

    #     else:
    #         raise NotImplementedError
