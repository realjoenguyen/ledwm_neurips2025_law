import os

from ledwm import jaxutils, ninjax as nj
from ledwm.nets import MLP

from ledwm.nets.Dist import TwoHotDist
import ledwm.Optimizer

# from ledwm.common import sg
from .jaxutils import sg
import jax.numpy as jnp


def fast_train_metrics_enabled():
    return os.environ.get("LEDWM_FAST_TRAIN_METRICS") == "1"


class VFunction(nj.Module):
    """
    for critic
    """

    def __init__(self, rewfn, config, slow_critic_target=None):
        # rewfn = lambda s: wm.heads["reward"](s).mean()[1:]
        self.rewfn = rewfn
        self.config = config
        self.net = MLP.MLP((), name="net", dims="deter", **self.config.critic)
        self.slow = MLP.MLP((), name="slow", dims="deter", **self.config.critic)
        self.updater = jaxutils.SlowUpdater(
            self.net,
            self.slow,
            self.config.slow_critic_fraction,
            self.config.slow_critic_update,
        )
        self.opt = ledwm.Optimizer.Optimizer(
            name="critic_opt", **self.config.critic_opt
        )
        if slow_critic_target is None:
            self.slow_critic_target = self.config.slow_critic_target

    def train(self, traj, actor, bs_bw=None, is_first=None, uncertainty=None):
        target = sg(
            self.score(traj, slow=self.slow_critic_target, uncertainty=uncertainty)[1]
        )
        mets, metrics = self.opt.__call__(
            self.net, self.critic_loss, traj, target, bs_bw, is_first, has_aux=True
        )
        metrics.update(mets)
        self.updater()
        return metrics

    def critic_loss(self, traj, target, bs_bw=None, is_first=None):  # horizon, bs * bl
        # print("critic loss")
        metrics = {}
        traj = {k: v[:-1] for k, v in traj.items()}
        if self.config.sg_ac:
            dist: "TwoHotDist" = self.net.__call__(sg(traj))
        else:
            dist: "TwoHotDist" = self.net.__call__(traj)

        loss = -dist.log_prob(sg(target))
        # regularization
        if self.config.critic_slowreg == "logprob":
            reg = -dist.log_prob(sg(self.slow(traj).mean()))

        elif self.config.critic_slowreg == "xent":
            reg = -jnp.einsum(
                "...i,...i->...", sg(self.slow(traj).probs), jnp.log(dist.probs)
            )
        else:
            raise NotImplementedError(self.config.critic_slowreg)

        loss += self.config.loss_scales.slowreg * reg
        # loss = (loss * sg(traj["weight"])).mean()
        loss = loss * sg(traj["weight"])
        loss *= self.config.loss_scales.critic  # horizon, bs * bl

        if bs_bw is not None:
            from ledwm.ActorCritic import scale_loss_bw

            assert self.config.replay.balanced_weight_ac
            loss = scale_loss_bw(loss, bs_bw)

        if not fast_train_metrics_enabled():
            metrics = jaxutils.tensorstats(dist.mean(), prefix="pred")
            metrics.update(jaxutils.tensorstats(target, prefix="target"))

        if is_first is not None:
            loss = loss * is_first[None, :]  # horizon, bs * bl (*) (1, bs * bl)
            loss = loss.sum() / (loss > 0).sum()  # mean over nonzero
            # loss
        else:
            loss = loss.mean()

        return loss, metrics

    def score(self, traj, actor=None, slow=False, uncertainty=None):
        # rewfn = lambda s: wm.heads["reward"](s).mean()[1:] - define in class Greedy
        rew = self.rewfn(traj)  # reward [1:] # horizon, bs * bl
        if uncertainty is not None:
            rew = jnp.maximum(
                self.config.min_reward,
                rew - uncertainty * self.config.uncertainty_scale,
            )
            # rew = rew - uncertainty * self.config.uncertainty_scale
            # rew = jnp.where(rew < -2, -2, rew)
        assert len(rew) == len(traj["action"]) - 1, (
            "should provide rewards for all but last action"
        )

        discount = 1 - 1 / self.config.horizon
        disc = traj["cont"][1:] * discount
        # horizon + 1,
        if slow:
            value = self.slow(traj).mean()
        else:
            value = self.net(traj).mean()

        vals = [value[-1]]
        interm = rew + disc * value[1:] * (1 - self.config.return_lambda)  # horizon,

        # horizon + 1
        for t in reversed(range(len(disc))):
            vals.append(interm[t] + disc[t] * self.config.return_lambda * vals[-1])

        ret = jnp.stack(list(reversed(vals))[:-1])
        return rew, ret, value[:-1]
