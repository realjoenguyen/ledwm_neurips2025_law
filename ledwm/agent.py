from termcolor import cprint

from ledwm.nets.EncoderSent import EncoderEmbed
from ledwm.WM import WM
from ledwm.embodied.core.counter import Counter
from ledwm.embodied.core.path import Path
from ledwm.embodied.replay import generic
from .jaxutils import (
    cast_to_compute,
    create_batch_images_from_pos,
    keep_is_first_only,
    tree_map,
)

import jax
import jax.numpy as jnp
import numpy as np
import logging
from ledwm.configs_util import load_configs

logger = logging.getLogger()


class CheckTypesFilter(logging.Filter):
    def filter(self, record):
        return "check_types" not in record.getMessage()


logger.addFilter(CheckTypesFilter())

from . import behaviors
from . import jaxagent
from . import ninjax as nj


@jaxagent.Wrapper
class Agent(nj.Module):
    configs = load_configs(Path(__file__).parent)

    def __init__(
        self,
        obs_space,
        act_space,
        step: "Counter",
        config,
        env_cache=None,
        reward_values=None,
    ):
        # from ledwm.embodied.envs.MessengerSent import MessengerSent

        self.config = config
        self.obs_space = obs_space
        self.act_space = act_space["action"]
        self.step = step
        self.env_cache = env_cache
        self.reward_values = reward_values

        with jax.transfer_guard("allow"):
            dummy_preproc = self.preprocess(
                {k: jnp.ones(v.shape) for k, v in self.obs_space.items()},
                reward_values=reward_values,
            )
            preproc_shapes = {
                k: tuple(v.shape)
                for k, v in dummy_preproc.items()
                if not k.startswith("log_")
            }

        self.wm = WM(
            obs_space,
            act_space,
            config,
            preproc_shapes,
            env_cache,
            reward_values,
            name="wm",  # type: ignore
        )

        self.preprocessors = {k: v() for k, v in self.wm.encoder.preprocessors.items()}

        # if self.config.run.pretrain_wm_only:
        # print("Agent: Pretraining WM only.")
        # return

        self.task_behavior: behaviors.Greedy = getattr(behaviors, config.task_behavior)(
            self.wm, self.act_space, self.config, name="task_behavior"
        )
        cprint(f"agent.behavior | role=task | type={config.task_behavior}")

        if config.expl_behavior == "None":
            self.expl_behavior = self.task_behavior
        else:
            self.expl_behavior = getattr(behaviors, config.expl_behavior)(
                self.wm, self.act_space, self.config, name="expl_behavior"
            )
            cprint(
                f"agent.behavior | role=exploration | type={config.expl_behavior}",
                "green",
            )

    def policy_initial(self, batch_size):
        print(f"agent.policy_state_init | batch_size={batch_size}")
        return (
            self.wm.initial(batch_size, use_for_policy=True),  # latent, action
            {},
            {},
            # self.task_behavior.initial(batch_size),  # {}
            # self.expl_behavior.initial(batch_size),  # {}
        )

    def train_initial(self, batch_size):
        # print("Agent: train_initial")
        # opt_step = nj.Variable(jnp.array, 0, jnp.int32, name="step")
        return self.wm.initial(batch_size)

    def policy(
        self,
        obs,
        prev_state,
        step=None,  # -1 if eval mode (eval, test)
        mode="train",  # test if parallel_eval
    ):
        self.config.jax.jit and cprint("jax.trace | function=policy")  # type: ignore
        obs = self.preprocess(obs)
        (prev_latent, prev_action), task_state, expl_state = prev_state
        assert prev_action.shape[1] > 0, prev_action.shape
        training = "train" in mode  # mode =train or training

        # from EncoderSent import EncoderEmbed
        # assert isinstance(self.wm.encoder, EncoderEmbed), self.wm.encoder
        obs_input = self.wm.encoder.__call__(obs, step, training)

        if self.config.rssm_type == "token":
            raise NotImplementedError
        else:
            if self.config.encoder_type == "token":
                # latent = self.wm.rssm.obs_step(
                #     prev_latent, prev_action, obs_input, obs["is_first"]
                # )
                raise NotImplementedError

            elif self.config.encoder_type == "sent":
                obs_input["action"] = prev_action
                obs_input["is_first"] = obs["is_first"]
                latent = self.wm.rssm.obs_step_sent(
                    step,
                    prev_latent,
                    data=obs_input,
                    training=training,
                    use_for_policy=True,
                )
            else:
                raise NotImplementedError(self.config.encoder_type)

        POLICY_LATENT_KEYS = ["stoch", "deter", "logit", "deter_layers"]
        latent = {k: v for k, v in latent.items() if k in POLICY_LATENT_KEYS}
        task_outs, task_state, info = self.task_behavior.policy(
            latent,
            task_state,
            step,
            sample=(
                self.config.measure_policy == "sample"
                if self.config.run.script == "finetune_policy"
                else training
            ),
        )

        # (expl_outs, expl_state, expl_info) = self.expl_behavior.policy(
        #     latent, expl_state, step
        # )
        outs = {
            "eval": task_outs,
            # "explore": expl_outs,
            "train": task_outs,
            "test": task_outs,
            "test-se": task_outs,
        }[mode]

        # reward pred and cont pred
        # reward_pred = self.wm.decoder_heads["reward"](latent, training=False)  # bs,
        # cont_pred = self.wm.decoder_heads["cont"](latent, training=False)  # bs,
        # outs["log_reward_pred"] = reward_pred.mode()  # type: ignore
        # outs["log_cont_pred"] = cont_pred.mode()  # type: ignore

        prev_state = ((latent, outs["action"]), task_state, expl_state)

        return outs, prev_state, info

    def train(
        self,
        data,
        state,
        step=None,
        last_reward_weights=None,
        reward_values=None,
    ):
        # state: tuple of
        # 'deter', 'logit', 'stoch' : bs, *
        # action
        # data: bs, bl
        if self.config.jax.jit:
            cprint("jax.trace | function=train")

        metrics = {}
        data = self.preprocess(data, reward_values)

        if self.config.train_policy_only:
            # obs_input = self.wm.encoder.__call__(data)
            # # obs_input = {k: v[:N_FIRST_BS, :N_FIRST_BL] for k, v in obs_input.items()}
            # context = self.wm.rssm.observe(
            #     obs_input,
            #     data["action"],
            #     data["is_first"],
            #     encoder_type="sent",
            #     step=step,
            #     training=False,
            #     verbose=True,  # take atten score
            # )
            # context = {**data, **context}
            # TODO: shift data['action'] by 1
            pass

        else:
            state, wm_outs, mets = self.wm.train(data, state, step, last_reward_weights)
            # state: ({'deter', 'stoch', 'logits'}, (bs, bl))
            # outs: all losses + 'prior' + 'post'
            metrics.update(mets)
            # wm_outs['post'] has 'deter' from RSSM.obs_step_sent
            context = {**data, **wm_outs["post"]}
            if self.config.run.pretrain_wm_only:
                return wm_outs, state, metrics

        # Flatten (bs, bl) -> (bs * bl)
        # context: data keys: 'is_first', 'is_last', 'deter', 'logit', etc.
        # context: bs, bl

        start = tree_map(lambda x: x.reshape((-1, *x.shape[2:])), context)

        _, mets = self.task_behavior.train(
            self.wm.imagine, start, context, step, last_reward_weights
        )
        assert isinstance(mets, dict), mets
        metrics.update(mets)

        if self.config.expl_behavior != "None":  # default is None - skip
            _, mets = self.expl_behavior.train(self.wm.imagine, start, context)
            metrics.update({"expl_" + key: value for key, value in mets.items()})
        outs = {}
        return outs, state, metrics

    def finetune_policy(
        self,
        data,  # bs, 1
        state,  # 'deter', 'logit', 'stoch' : bs, *
        step=None,
        uncertainty=None,
    ):
        if self.config.jax.jit:
            cprint("jax.trace | function=finetune_policy", "green")

        metrics = {}
        data = self.preprocess(data)
        # prev_latent, prev_action = state
        # assert isinstance(self.wm.encoder, EncoderEmbed)
        obs_input = self.wm.encoder.__call__(data, training=False)

        if self.config.rssm_type == "token":
            raise NotImplementedError
        else:
            assert obs_input is not None, "Embedding is None"
            post = self.wm.rssm.observe(
                obs_input,  # bs, bl
                data["action"],
                data["is_first"],
                # prev_latent,
                step=step,  # in case of soft-z
                encoder_type=self.config.encoder_type,
                training=False,
            )  # after observing the first obs

        context = {**data, **post}
        # Flatten (bs, bl) -> (bs * bl)
        # context: data keys: 'is_first', 'is_last', 'deter', 'logit', etc.
        # context: bs, bl
        start = tree_map(lambda x: x.reshape((-1, *x.shape[2:])), context)  # bs * bl
        # type: ActorCritic.train
        _, mets = self.task_behavior.train(
            # self.wm.imagine,
            lambda *args: self.wm.imagine(*args, training=False),
            start,
            context,
            uncertainty=uncertainty,
        )
        assert isinstance(metrics, dict), metrics
        metrics.update(mets)

        outs = {}
        return outs, state, metrics

    # def finetune_policy(
    #     self,
    #     data,  # bs, 1
    #     state,  # 'deter', 'logit', 'stoch' : bs, *
    #     step=None,
    #     grad_step=1,  # New parameter: number of gradient steps
    # ):
    #     if self.config.jax.jit:
    #         cprint("Tracing finetune_policy function.", "green")

    #     data = self.preprocess(data)
    #     obs_input = self.wm.encoder.__call__(data, training=False)

    #     if self.config.rssm_type == "token":
    #         raise NotImplementedError
    #     else:
    #         assert obs_input is not None, "Embedding is None"
    #         post = self.wm.rssm.observe(
    #             obs_input,  # bs, bl
    #             data["action"],
    #             data["is_first"],
    #             step=step,  # in case of soft-z
    #             encoder_type=self.config.encoder_type,
    #             training=False,
    #         )  # after observing the first obs

    #     context = {**data, **post}
    #     start = tree_map(
    #         lambda x: x.reshape((-1, *x.shape[2:])), context
    #     )  # Flatten (bs, bl) -> (bs * bl)

    #     def train_step(carry, i):
    #         state = carry
    #         _, _ = self.task_behavior.train(
    #             lambda *args: self.wm.imagine(*args, training=False),
    #             start,
    #             context,
    #         )
    #         return state, None

    #     # Initialize state as the carry
    #     init_carry = state

    #     # Run the loop
    #     final_state, _ = jax.lax.fori_loop(0, grad_step, train_step, init_carry)

    #     outs = {}
    #     return outs, final_state, {}

    def train_wm(self, data, state, step=None):
        metrics = {}
        data = self.preprocess(data)
        state, wm_outs, mets = self.wm.train(data, state, step)
        metrics.update(mets)
        context = {**data, **wm_outs["post"]}
        return wm_outs, state, metrics

    def report_policy(
        self,
        data,  # has the first action: reset
        step=None,
    ):
        """
        the policy is in self.task_behavior
        """
        self.config.jax.jit and cprint("Tracing report_policy function.")  # type: ignore
        data = self.preprocess(data, keep_keys=generic.KEEP_KEYS)
        report = {}
        num_eps = self.config.num_eval_eps

        # assert isinstance(self.wm.encoder, EncoderEmbed)
        obs_input = self.wm.encoder.__call__(data, training=False)
        context = self.wm.rssm.observe(
            obs_input,
            data["action"],
            data["is_first"],
            encoder_type="sent",
            training=False,
            verbose=True,  # take atten score
        )
        context = {**data, **context}
        start = tree_map(lambda x: x.reshape((-1, *x.shape[2:])), context)  # bs * bl
        assert start["reward"].shape[0] == num_eps, (
            f"{start['reward'].shape[0]=} != {num_eps=}"
        )

        def test_policy(*args_fn):
            return self.task_behavior.policy(
                *args_fn, sample=self.config.measure_policy == "sample"
            )

        traj = self.wm.imagine(
            test_policy,
            start,
            self.config.imag_horizon,
            step=step,
            carry={},
            training=False,
        )  # horizon + 1, bs, d*

        if self.config.reward_head.dist == "onehot":

            def rewfn(s):
                return self.wm.decoder_heads["reward"](s, training=False).mode(
                    self.wm.reward_values
                )
        else:

            def rewfn(s):
                return self.wm.decoder_heads["reward"](s, training=False).mode()

        sum_reward = rewfn(traj).sum() / num_eps
        report["sum_reward"] = sum_reward
        report["reward"] = rewfn(traj)  # horizon + 1, bs
        report["cont"] = traj["cont"]  # horizon + 1, bs
        assert report["reward"].shape == report["cont"].shape, (
            report["reward"].shape,
            report["cont"].shape,
        )
        return report

    def report(self, data, step=None):
        self.config.jax.jit and cprint("Tracing report function.", "green")
        data = self.preprocess(data, keep_keys=generic.KEEP_KEYS)
        report = {}
        # if self.config.drop_x_randomly_in_query and training and "drop_x" not in data:
        #     bs = data["action"].shape[0]
        #     drop_x = jax.random.bernoulli(nj.rng(), 0.5, (bs,))
        #     data["drop_x"] = drop_x

        report.update(self.wm.report(data, step))
        if self.config.run.pretrain_wm_only:
            return report

        mets = self.task_behavior.report(data, step)
        report.update({f"task_{k}": v for k, v in mets.items()})

        if self.expl_behavior is not self.task_behavior:
            mets = self.expl_behavior.report(data)
            report.update({f"expl_{k}": v for k, v in mets.items()})
        return report

    def vis(self, data, num_obs, num_imagine):
        data = self.preprocess(data)
        return self.wm.vis(data, num_obs, num_imagine)

    def save(self):
        data = jax.tree_util.tree_flatten(
            jax.tree_util.tree_map(jnp.asarray, self.state)
        )[0]
        data = [np.asarray(x) for x in data]
        return data

    def load(self, state):
        self.state = jax.tree_util.tree_flatten(self.state)[1].unflatten(state)

    def preprocess(self, batch, keep_keys=None, reward_values=None):
        """
        batch: can be (bs, bl, d*) or just one instance (d*) or (bl, d*)
        """

        batch = batch.copy()
        for key, value in batch.items():
            if key.startswith("log_"):
                if keep_keys is not None:
                    if key not in keep_keys:
                        continue
                else:
                    continue

            # for ids and pos, convert to int32 for embedding
            keep_this_key = key.endswith("_ids") or key.endswith("_pos")
            if self.config.use_read or self.config.use_time:
                keep_this_key = keep_this_key or key.endswith("_step")

            if keep_this_key:
                value = value.astype(jnp.int32)

            elif key == "token":
                value = jax.nn.one_hot(value, self.obs_space[key].high)

            elif (
                len(value.shape) > 3
                and value.dtype == jnp.uint8
                and key != "lang_image"
            ):
                cprint(
                    f"agent.input_cast | key={key} | dtype=float | scale=1/255",
                    "yellow",
                )
                value = cast_to_compute(value) / 255.0

            else:
                value = value.astype(jnp.float32)
            batch[key] = value

        # 1 means not terminal
        batch["cont"] = 1.0 - batch["is_terminal"].astype(jnp.float32)

        # make image from ids and pos
        if "image" not in batch and self.config.has_image_input:
            batch["image"] = create_batch_images_from_pos(
                batch["entity_pos"],
                batch["avatar_pos"],
                *self.config.env.messenger.size,
            )

        # if self.config.multi_step:
        #     batch["start_id"], batch["end_id"] = find_starts_ends(
        #         batch["is_first"],
        #         batch["is_last"],
        #         get_min_eps_len(self.config),
        #         get_max_eps_len(self.config),
        #     )  # bs, max_eps; bs, max_eps
        #     # batch["start_id"] = cast_to_compute(batch["start_id"])
        #     # batch["end_id"] = cast_to_compute(batch["end_id"])

        if self.config.reward_head.dist == "onehot":
            # with jax.transfer_guard("allow"):
            #     REWARD_VALUES = jnp.array([-2, -1, 0, 0.5, 1])
            reward_values = (
                reward_values if reward_values is not None else self.reward_values
            )
            assert reward_values is not None, "Reward values not set."
            if len(reward_values.shape) == 2:
                reward_values = reward_values[0]
            batch["reward"] = reward_real_to_onehot(batch["reward"], reward_values)
        return batch


# %%
def reward_real_to_onehot(rewards_real, reward_values):
    indices = jnp.argmin(jnp.abs(rewards_real[..., None] - reward_values), axis=-1)
    assert indices.shape == rewards_real.shape
    num_classes = reward_values.shape[0]
    return jax.nn.one_hot(indices, num_classes)


def test_reward_real_to_onehot():
    import jax
    import jax.numpy as jnp

    # Reward values
    REWARD_VALUES = jnp.array([-2, -1, 0, 0.5, 1])  # Shape (5,)

    # Example batch of rewards
    batch = {"reward": jnp.array([0, 1, -1, 0.5, 0])}  # Shape (6,)

    one_hot_rewards = reward_real_to_onehot(batch["reward"], REWARD_VALUES)
    print("Batch Rewards:", batch["reward"])
    print("Onehot", one_hot_rewards)
    # assert
    expected_onehot = jnp.array(
        [
            [0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1],
            [0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0],
        ]
    )
    assert jnp.allclose(one_hot_rewards, expected_onehot), (
        "One-hot encoding is incorrect"
    )
    from termcolor import cprint

    cprint("Test passed", "green")


if __name__ == "__main__":
    import jax
    import jax.numpy as jnp

    test_reward_real_to_onehot()
