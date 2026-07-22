import jax.numpy as jnp


import ledwm.ActorCritic
import ledwm.tfp_compat  # noqa: F401
from tensorflow_probability.substrates.jax import distributions as tfd

from . import VFunction
from . import expl
from . import ninjax as nj
from ledwm.nets import Dist


class Greedy(nj.Module):
    def __init__(self, wm, act_space, config):
        if config.reward_head.dist == "onehot":

            def rewfn(s):
                reward_reals = wm.decoder_heads["reward"](s, training=False).mode(
                    wm.reward_values
                )[1:]
                return reward_reals
        else:

            def rewfn(s):
                return wm.decoder_heads["reward"](s, training=False).mean()[1:]

        self.wm = wm
        if config.critic_type == "vfunction":
            critics = {"extr": VFunction.VFunction(rewfn, config, name="critic")}
        else:
            raise NotImplementedError(config.critic_type)
        self.ac = ledwm.ActorCritic.ActorCritic(
            critics, {"extr": 1.0}, act_space, config, name="ac"
        )

    def initial(self, batch_size):
        return self.ac.initial(batch_size)

    def policy(self, latent, state, step=None, sample=True):
        return self.ac.policy(latent, state, step=step, sample=sample)

    def train(
        self,
        imagine,
        start,
        data,
        step=None,
        last_reward_weights=None,
        uncertainty=None,
    ):
        return self.ac.train(
            imagine, start, data, step, last_reward_weights, uncertainty
        )

    def report(
        self,
        data,  # has action
        step=None,
        training=False,
    ):
        return {}
        # state = self.initial(len(data["is_first"]))
        # report = {}
        # obs_input = self.wm.encoder.__call__(data, training=training)
        # context = self.wm.rssm.observe(
        #     obs_input,
        #     data["action"],
        #     data["is_first"],
        #     encoder_type="sent",
        #     step=step,
        #     training=training,
        #     verbose=True,  # take atten score
        # )


class Random(nj.Module):
    def __init__(self, wm, act_space, config):
        self.config = config
        self.act_space = act_space

    def initial(self, batch_size):
        return jnp.zeros(batch_size)

    def policy(self, latent, state, step=None, sample=True):
        batch_size = latent["deter"].shape[0]
        assert batch_size > 0, batch_size
        shape = (batch_size,) + self.act_space.shape
        if self.act_space.discrete:
            dist = Dist.OneHotDist(jnp.zeros(shape))
        else:
            dist = tfd.Uniform(-jnp.ones(shape), jnp.ones(shape))
            dist = tfd.Independent(dist, 1)

        action = dist.sample(seed=nj.rng())
        return {"action": action}, state

    def train(self, imagine, start, data, step=None, last_reward_weights=None):
        return None, {}

    def report(self, data, step):
        return {}


class Explore(nj.Module):
    pass
