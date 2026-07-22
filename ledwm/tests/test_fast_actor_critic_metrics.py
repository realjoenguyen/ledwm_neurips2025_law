import os
import unittest
from types import SimpleNamespace
from unittest import mock

import jax.numpy as jnp

from ledwm import ActorCritic as actor_critic_module
from ledwm import VFunction as value_function_module


class _ActorDistribution:
    def log_prob(self, value):
        return jnp.zeros(value.shape[:-1], jnp.float32)

    def entropy(self):
        return jnp.zeros((3, 2), jnp.float32)


class _ActorNetwork:
    def __call__(self, value, *args, **kwargs):
        del value, args, kwargs
        return _ActorDistribution()


class _ValueDistribution:
    def log_prob(self, value):
        return jnp.zeros_like(value)

    def mean(self):
        return jnp.zeros((2, 2), jnp.float32)


class _ValueNetwork:
    def __call__(self, value, *args, **kwargs):
        del value, args, kwargs
        return _ValueDistribution()


class FastActorCriticMetricsTest(unittest.TestCase):
    def setUp(self):
        self.old_flag = os.environ.get("LEDWM_FAST_TRAIN_METRICS")
        os.environ["LEDWM_FAST_TRAIN_METRICS"] = "1"

    def tearDown(self):
        if self.old_flag is None:
            os.environ.pop("LEDWM_FAST_TRAIN_METRICS", None)
        else:
            os.environ["LEDWM_FAST_TRAIN_METRICS"] = self.old_flag

    def test_actor_loss_skips_diagnostic_metrics(self):
        class _Critic:
            def score(self, traj, actor, uncertainty=None):
                del traj, actor, uncertainty
                values = jnp.zeros((2, 2), jnp.float32)
                return values, values, values

        class _ReturnNorm:
            def __call__(self, value):
                return jnp.zeros_like(value), jnp.ones_like(value)

        actor_critic = SimpleNamespace(
            critics={"extr": _Critic()},
            scales={"extr": 1.0},
            retnorms={"extr": _ReturnNorm()},
            actor=_ActorNetwork(),
            grad="reinforce",
            config=SimpleNamespace(
                actor_entropy=0.0,
                loss_scales=SimpleNamespace(actor=1.0),
                replay=SimpleNamespace(balanced_weight_ac=False),
            ),
            metrics=mock.Mock(side_effect=AssertionError("diagnostic metrics ran")),
        )
        traj = {
            "action": jnp.zeros((3, 2, 2), jnp.float32),
            "weight": jnp.ones((3, 2), jnp.float32),
        }

        with mock.patch.object(
            actor_critic_module.jaxutils,
            "tensorstats",
            side_effect=AssertionError("tensorstats ran"),
        ):
            loss, metrics = actor_critic_module.ActorCritic.actor_loss.__wrapped__(
                actor_critic, traj
            )

        self.assertEqual(loss.shape, ())
        self.assertEqual(metrics, {})
        actor_critic.metrics.assert_not_called()

    def test_critic_loss_skips_diagnostic_metrics(self):
        critic = SimpleNamespace(
            config=SimpleNamespace(
                sg_ac=True,
                critic_slowreg="logprob",
                loss_scales=SimpleNamespace(slowreg=0.1, critic=1.0),
                replay=SimpleNamespace(balanced_weight_ac=False),
            ),
            net=_ValueNetwork(),
            slow=_ValueNetwork(),
        )
        traj = {
            "feature": jnp.zeros((3, 2, 1), jnp.float32),
            "weight": jnp.ones((3, 2), jnp.float32),
        }
        target = jnp.zeros((2, 2), jnp.float32)

        with mock.patch.object(
            value_function_module.jaxutils,
            "tensorstats",
            side_effect=AssertionError("tensorstats ran"),
        ):
            loss, metrics = value_function_module.VFunction.critic_loss.__wrapped__(
                critic, traj, target
            )

        self.assertEqual(loss.shape, ())
        self.assertEqual(metrics, {})


if __name__ == "__main__":
    unittest.main()
