from ledwm import jaxutils, ninjax as nj
from ledwm.nets import MLP
from ledwm.nets.Dist import TwoHotDist
import ledwm.Optimizer
from .jaxutils import sg
import jax.numpy as jnp
import jax

VALUE_WM_KEYS = ["is_terminal", "reward", "action", "deter", "stoch", "deter_layers"]


class VFunction(nj.Module):
    """
    value prediction for wm
    """

    def __init__(self, config):
        # rewfn = lambda obs: obs["reward"][1:]
        # self.rewfn = rewfn
        self.config = config
        self.net = MLP.MLP((), name="net", dims="deter", **self.config.value_head)
        self.slow = MLP.MLP((), name="slow", dims="deter", **self.config.value_head)
        self.updater = jaxutils.SlowUpdater(
            self.net,
            self.slow,
            self.config.slow_value_fraction,
            self.config.slow_value_update,
        )

    # def train(self, traj, actor, bs_bw=None):
    #     target = sg(self.score(traj, slow=self.slow_critic_target)[1])
    #     mets, metrics = self.opt.__call__(
    #         self.net, self.critic_loss, traj, target, bs_bw, has_aux=True
    #     )
    #     metrics.update(mets)
    #     self.updater()
    #     return metrics

    def target(self, data):
        return sg(self.score(data, slow=self.config.slow_value_target)[1])

    def dist(self, data):
        return self.net.__call__(data)

    def critic_loss(
        self,
        data,  # bs, bl,d*
        target=None,
    ):
        # print("critic loss")
        metrics = {}

        data = {k: v for k, v in data.items() if k in VALUE_WM_KEYS}  # bs, bl, d*
        # turn bs, bl, d* into bl, bs, d*. By swap axis 0 and 1
        data = {k: v.swapaxes(0, 1) for k, v in data.items()}
        if target is None:
            target = self.target(data)
        data = {k: v[:-1] for k, v in data.items()}

        # swap axis 0 and 1 back to bs, bl, d*
        target = target.swapaxes(0, 1)
        data = {k: v.swapaxes(0, 1) for k, v in data.items()}
        # if self.config.sg_ac:
        #     dist: "TwoHotDist" = self.net.__call__(sg(traj))
        # else:
        dist: "TwoHotDist" = self.net.__call__(data)
        loss = -dist.log_prob(sg(target))
        # regularization
        if self.config.critic_slowreg == "logprob":
            reg = -dist.log_prob(sg(self.slow(data).mean()))

        loss += self.config.loss_scales.slowreg * reg
        # loss = (loss * sg(traj["weight"])).mean()

        discount = 1 - 1 / self.config.horizon
        cont = 1.0 - data["is_terminal"].astype(jnp.float32)  # 1 means not terminal
        weights = jnp.cumprod(discount * cont, 0) / discount

        loss = loss * sg(weights)
        loss *= self.config.loss_scales.value  # horizon, bs * bl

        metrics = jaxutils.tensorstats(dist.mean(), prefix="value_pred")
        metrics.update(jaxutils.tensorstats(target, prefix="value_target"))
        metrics["value_kl_mean"] = dist.kl(target).mean()

        return dist, loss, metrics

    def score(self, data, slow=False):
        rewards = data["reward"][1:]  # bl, bs, d*
        # jax.debug.print("rewards={rewards}, {rewards.shape}", rewards=rewards)
        cont = 1.0 - data["is_terminal"].astype(jnp.float32)  # 1 means not terminal
        assert len(rewards) == len(data["action"]) - 1, (
            "should provide rewards for all but last action"
        )

        discount = 1 - 1 / self.config.horizon
        disc = cont[1:] * discount
        if slow:
            value = self.slow(data).mean()
        else:
            value = self.net(data).mean()

        vals = [value[-1]]
        interm = rewards + disc * value[1:] * (1 - self.config.return_lambda)

        for t in reversed(range(len(disc))):
            vals.append(interm[t] + disc[t] * self.config.return_lambda * vals[-1])

        returns = jnp.stack(list(reversed(vals))[:-1])
        # jax.debug.print("returns={returns}, {returns.shape}", returns=returns)
        # jax.debug.print("rewards={rewards}, {rewards.shape}", rewards=rewards)
        return rewards, returns, value[:-1]
