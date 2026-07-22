import os
from typing import TYPE_CHECKING, Callable, Dict

import jax
from termcolor import cprint
from ledwm import jaxutils, ninjax as nj
from ledwm.nets import MLP
import ledwm.Optimizer
from ledwm.VFunction import VFunction


if TYPE_CHECKING:
    from ledwm.WM import WM


# from ledwm.common import sg
from .jaxutils import sg
import jax.numpy as jnp


def fast_train_metrics_enabled():
    return os.environ.get("LEDWM_FAST_TRAIN_METRICS") == "1"


def scale_loss_bw(
    loss,  # h, bs * bl
    bs_bw,  # bs,
):
    """

    Args:
        loss (_type_): _description_
        bs (_type_): _description_

    Returns:
        (horizon, bs * bl)
    """
    h, bsbl = loss.shape
    bs = bs_bw.shape[0]
    loss = loss.reshape((h, bs, bsbl // bs))
    loss = loss * bs_bw[None, :, None]
    return loss.reshape((h, bsbl))


class ActorCritic(nj.Module):
    def __init__(self, critics: "Dict[str, VFunction]", scales, act_space, config):
        critics = {k: v for k, v in critics.items() if scales[k]}
        for key, scale in scales.items():
            assert not scale or key in critics, key
        self.critics = {k: v for k, v in critics.items() if scales[k]}
        self.scales = scales
        self.act_space = act_space
        self.config = config
        disc = act_space.discrete
        self.grad = config.actor_grad_disc if disc else config.actor_grad_cont
        cprint(f"actor_critic.config | gradient={self.grad}")
        self.norm_adv = config.norm_adv

        self.actor = MLP.MLP(
            name="actor",
            dims="deter",
            shape=act_space.shape,
            **config.actor,
            dist=config.actor_dist_disc if disc else config.actor_dist_cont,
        )

        self.retnorms = {
            k: jaxutils.Moments(**config.retnorm, name=f"retnorm_{k}") for k in critics
        }
        if config.norm_adv:
            self.advnorm = jaxutils.Moments(**config.advnorm, name=f"advnorm")

        self.opt = ledwm.Optimizer.Optimizer(name="actor_opt", **config.actor_opt)
        # self.count_step = 0

    def initial(self):
        return {}

    def reset_opt(self):
        self.opt.reset()

    # def report(self, data, step=None):
    #     dist = self.actor.__call__(sg(state), step)
    #     metrics = {}
    #     ent = policy.entropy()[:-1]
    #     rand = (ent - policy.minent) / (policy.maxent - policy.minent)
    #     rand = rand.mean(range(2, len(rand.shape)))
    #     act = traj["action"]
    #     act = jnp.argmax(act, -1) if self.act_space.discrete else act
    #     act_max_prob = jnp.max(policy.probs, axis=-1)
    #     bs_bl = policy.probs.shape[1]

    #     # action_dim = policy.probs.shape[-1]
    #     horizon = policy.probs.shape[0]

    #     if step is not None:
    #         bs = step.shape[0]
    #         # step: bs -> repeat to bs, bl -> reshape to bs * bl
    #         step_repeat = step[:, None].repeat(bs_bl // bs, axis=1).reshape(-1)
    #         # step: from bs*bl repeat to horizon, bs*bl
    #         step_repeat = step_repeat[None, :].repeat(horizon, axis=0)
    #         # repeat to horizon, bs*bl, 5
    #         # step_repeat = step_repeat[:, :, None].repeat(action_dim, axis=-1)
    #         is_train_step = step_repeat != -1
    #         is_eval_step = step_repeat == -1
    #     else:
    #         is_train_step = None
    #         is_eval_step = None

    #     metrics.update(jaxutils.tensorstats(act, "action"))
    #     if is_train_step is not None and is_eval_step is not None:
    #         metrics.update(
    #             jaxutils.tensorstats(
    #                 act_max_prob, "train/max_action_prob", is_train_step
    #             )
    #         )
    #         metrics.update(
    #             jaxutils.tensorstats(act_max_prob, "eval/max_action_prob", is_eval_step)
    #         )
    #     else:
    #         metrics.update(jaxutils.tensorstats(act_max_prob, "max_action_prob"))

    #     metrics.update(jaxutils.tensorstats(rand, "policy_randomness"))
    #     if is_train_step is not None and is_eval_step is not None:
    #         metrics.update(
    #             jaxutils.tensorstats(
    #                 ent,
    #                 "train/policy_entropy",
    #                 is_train_step[:-1] if is_train_step is not None else None,
    #             )
    #         )
    #         metrics.update(
    #             jaxutils.tensorstats(
    #                 ent,
    #                 "eval/policy_entropy",
    #                 is_eval_step[:-1] if is_eval_step is not None else None,
    #             )
    #         )
    #     else:
    #         metrics.update(jaxutils.tensorstats(ent, "policy_entropy"))

    #     metrics.update(jaxutils.tensorstats(logpi, "policy_logprob"))

    #     if is_train_step is not None and is_eval_step is not None:
    #         metrics.update(
    #             jaxutils.tensorstats(
    #                 adv,
    #                 "train/adv",
    #                 is_train_step[:-1] if is_train_step is not None else None,
    #             )
    #         )
    #         metrics.update(
    #             jaxutils.tensorstats(
    #                 adv,
    #                 "eval/adv",
    #                 is_eval_step[:-1] if is_eval_step is not None else None,
    #             )
    #         )
    #     else:
    #         metrics.update(jaxutils.tensorstats(adv, "adv"))

    #     metrics["imag_weight_dist"] = jaxutils.subsample(traj["weight"])
    #     return metrics

    def policy(self, state, carry, step=None, sample=True):
        if self.config.finetune_script == "measure_return":
            sample = False

        if self.config.sg_ac:
            state = sg(state)

        dist = self.actor.__call__(state, step, training=sample)
        if sample:
            if step is not None:
                if step.shape[0] != dist.batch_shape[0]:
                    step = step[0, None].repeat(dist.batch_shape[0], axis=0).reshape(-1)

                step_broadcast = step.reshape(
                    step.shape[0], *([1] * (len(dist.event_shape)))
                )
                action = jnp.where(
                    step_broadcast == -1,
                    dist.mode(),  # eval + test
                    dist.sample(
                        seed=nj.rng(),
                    ),  # train
                )
            else:
                action = dist.sample(seed=nj.rng())
        else:
            # jax.debug.print("policy={policy}", policy=dist.probs)
            action = dist.mode()

        # info = {"probs": dist.probs, "step": step, "sample": sample}
        info = {}
        return {"action": action}, carry, info

    def train(
        self,
        imagine: "Callable[WM.imagine]",
        start,  # bs * bl: only need: deter, logit, stoch
        context,  # bs, bl: not sure what it is, you don't need context. Only use this for bs, bl
        step=None,
        last_reward_weights=None,
        uncertainty=None,
    ):
        from ledwm.WM import get_bs_bw_weights

        # bs, bl = context["action"].shape[:2]
        carry = self.initial()
        use_bw = (
            last_reward_weights is not None and self.config.replay.balanced_weight_ac
        )
        if use_bw:
            bs_bw_weights = get_bs_bw_weights(context, last_reward_weights)  # bs,
        else:
            bs_bw_weights = None

        # ACTOR LOSS
        def loss(start):
            # horizon, bs * bl, *d
            traj = imagine(self.policy, start, self.config.imag_horizon, carry, step)

            loss, metrics = self.actor_loss(
                traj,
                step,
                bs_bw=bs_bw_weights,
                is_first=(
                    start["is_first"] if self.config.train_policy_is_first else None
                ),
                uncertainty=uncertainty,
            )
            return loss, (traj, metrics)

        mets, (traj, metrics) = self.opt.__call__(self.actor, loss, start, has_aux=True)
        metrics.update(mets)

        # CRITIC LOSS
        for key, critic in self.critics.items():
            critic: VFunction
            mets = critic.train(
                traj,
                self.actor,
                bs_bw=bs_bw_weights,
                is_first=(
                    start["is_first"] if self.config.train_policy_is_first else None
                ),
                uncertainty=uncertainty,
            )
            metrics.update({f"{key}_critic_{k}": v for k, v in mets.items()})
        return traj, metrics

    def actor_loss(
        self,
        traj,  # horizon, bs * bl, *d
        step=None,  # int
        bs_bw=None,  # bs,
        is_first=None,  # bs*bl
        uncertainty=None,
    ):
        # ACTOR LOSS
        metrics = {}
        fast_metrics = fast_train_metrics_enabled()
        advs = []
        total = sum(self.scales[k] for k in self.critics)

        for key, critic in self.critics.items():
            critic: VFunction
            rew, ret, base = critic.score(
                traj, self.actor, uncertainty=uncertainty
            )  # horizon, *
            offset, invscale = self.retnorms[key].__call__(ret)  # offset: low
            normed_ret = (ret - offset) / invscale
            normed_base = (base - offset) / invscale
            advs.append((normed_ret - normed_base) * self.scales[key] / total)

            if not fast_metrics:
                metrics.update(jaxutils.tensorstats(offset, f"{key}_offset"))
                metrics.update(jaxutils.tensorstats(invscale, f"{key}_invscale"))
                metrics.update(jaxutils.tensorstats(rew, f"{key}_reward"))
                metrics.update(jaxutils.tensorstats(ret, f"{key}_return_raw"))
                metrics.update(
                    jaxutils.tensorstats(normed_ret, f"{key}_return_normed")
                )
                metrics.update(
                    jaxutils.tensorstats(normed_base, f"{key}_based_normed")
                )
                metrics[f"{key}_abs_return_rate"] = (jnp.abs(ret) >= 0.5).mean()
                metrics[f"{key}_return_rate"] = (ret >= 0.5).sum() / ret.shape[0]
                metrics[f"{key}_reward_rate"] = (rew >= 0.5).mean() / ret.shape[0]
                metrics[f"{key}_reward_rate_0.1"] = (rew >= 0.1).mean() / ret.shape[0]
                metrics[f"{key}_reward_neg_rate"] = (rew <= -0.1).mean() / ret.shape[0]

        adv = jnp.stack(advs).sum(0)
        policy = self.actor.__call__(sg(traj), step)
        logpi = policy.log_prob(sg(traj["action"]))[:-1]

        # horizon (s1=4), bs * bl
        # if self.norm_adv:
        #     offset, invscale = self.advnorm(adv)
        #     normed_adv = (adv - offset) / invscale
        #     metrics.update(jaxutils.tensorstats(normed_adv, f"adv_normed"))
        #     loss = {"backprop": -adv, "reinforce": -logpi * sg(normed_adv)}[self.grad]
        # else:
        if not fast_metrics:
            metrics.update(jaxutils.tensorstats(adv, "adv"))
        loss = {"backprop": -adv, "reinforce": -logpi * sg(adv)}[self.grad]
        if not fast_metrics:
            metrics.update(jaxutils.tensorstats(-logpi * sg(adv), "reinforce_loss"))

        ent = policy.entropy()[:-1]
        # self.count_step += ent.shape[0] * ent.shape[1]
        loss -= self.config.actor_entropy * ent
        if not fast_metrics:
            metrics.update(
                jaxutils.tensorstats(self.config.actor_entropy * ent, "entropy_loss")
            )

        # expotential entropy decay from T=0 to T=5e2: go from 3e-2 to self.config.actor_entropy:
        # actor_entropy = 3e-2 * jnp.exp(self.count_step * (-jnp.log(3e-2) / 5e2))
        # loss -= actor_entropy * ent
        loss *= sg(traj["weight"])[:-1]
        if not fast_metrics:
            metrics.update(jaxutils.tensorstats(sg(traj["weight"])[:-1], "weight"))
        loss *= self.config.loss_scales.actor  # horizon, bs * bl
        if not fast_metrics:
            metrics.update(jaxutils.tensorstats(loss, "actor_loss"))

        if bs_bw is not None:  # bs,
            assert self.config.replay.balanced_weight_ac
            loss = scale_loss_bw(loss, bs_bw)

        if not fast_metrics:
            metrics.update(self.metrics(traj, policy, logpi, ent, adv, step))

        if is_first is not None:
            loss = loss * is_first[None, :]  # horizon, bs * bl (*) (1, bs * bl)
            loss = loss.sum() / (loss > 0).sum()  # mean over nonzero
        else:
            loss = loss.mean()
        return loss, metrics

    def metrics(
        self,
        traj,
        policy,  # self.actor.__call__(sg(traj), step): horizon, bs * bl, 5
        logpi,
        ent,
        adv,
        step=None,  # bs,
    ):
        metrics = {}
        ent = policy.entropy()[:-1]
        rand = (ent - policy.minent) / (policy.maxent - policy.minent)
        rand = rand.mean(range(2, len(rand.shape)))
        act = traj["action"]
        act = jnp.argmax(act, -1) if self.act_space.discrete else act
        act_max_prob = jnp.max(policy.probs, axis=-1)
        bs_bl = policy.probs.shape[1]

        # action_dim = policy.probs.shape[-1]
        horizon = policy.probs.shape[0]

        if step is not None:
            bs = step.shape[0]
            # step: bs -> repeat to bs, bl -> reshape to bs * bl
            step_repeat = step[0, None].repeat(bs_bl, axis=0).reshape(-1)
            # step: from bs*bl repeat to horizon, bs*bl
            step_repeat = step_repeat[None, :].repeat(horizon, axis=0)
            # repeat to horizon, bs*bl, 5
            # step_repeat = step_repeat[:, :, None].repeat(action_dim, axis=-1)
            is_train_step = step_repeat != -1
            is_eval_step = step_repeat == -1
        else:
            is_train_step = None
            is_eval_step = None

        metrics.update(jaxutils.tensorstats(act, "action"))
        if is_train_step is not None and is_eval_step is not None:
            metrics.update(
                jaxutils.tensorstats(
                    act_max_prob, "train/max_action_prob", is_train_step
                )
            )
            metrics.update(
                jaxutils.tensorstats(act_max_prob, "eval/max_action_prob", is_eval_step)
            )
        else:
            metrics.update(jaxutils.tensorstats(act_max_prob, "max_action_prob"))

        metrics.update(jaxutils.tensorstats(rand, "policy_randomness"))
        if is_train_step is not None and is_eval_step is not None:
            metrics.update(
                jaxutils.tensorstats(
                    ent,
                    "train/policy_entropy",
                    is_train_step[:-1] if is_train_step is not None else None,
                )
            )
            metrics.update(
                jaxutils.tensorstats(
                    ent,
                    "eval/policy_entropy",
                    is_eval_step[:-1] if is_eval_step is not None else None,
                )
            )
        else:
            metrics.update(jaxutils.tensorstats(ent, "policy_entropy"))

        metrics.update(jaxutils.tensorstats(logpi, "policy_logprob"))

        if is_train_step is not None and is_eval_step is not None:
            metrics.update(
                jaxutils.tensorstats(
                    adv,
                    "train/adv",
                    is_train_step[:-1] if is_train_step is not None else None,
                )
            )
            metrics.update(
                jaxutils.tensorstats(
                    adv,
                    "eval/adv",
                    is_eval_step[:-1] if is_eval_step is not None else None,
                )
            )
        else:
            metrics.update(jaxutils.tensorstats(adv, "adv"))

        metrics["imag_weight_dist"] = jaxutils.subsample(traj["weight"])
        return metrics
