import os
import pathlib
import sys
import unittest
from types import SimpleNamespace

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "messenger-emma"))

from ledwm.Optimizer import (  # noqa: E402
    fast_optimizer_metrics_enabled,
    skip_adam_metrics_enabled,
)
from ledwm.jaxagent import JAXAgent, skip_train_outs_enabled  # noqa: E402
from ledwm.train import (  # noqa: E402
    configure_training_speed_flags,
    warmup_agent_for_capacity_probe,
)


class PromiseFake:
    def __init__(self):
        self.result_called = False

    def result(self):
        self.result_called = True
        return {"previous": True}


class WorkerFake:
    def __init__(self):
        self.submitted = []

    def submit(self, fn, *args):
        self.submitted.append((fn, args))
        return PromiseFake()


class AgentFake:
    def __init__(self):
        self.outs_promise = None
        self.outs_worker = WorkerFake()
        self.train_devices = ("train0",)

    def _convert_outs(self, outs, devices):
        return outs, devices


class CapacityProbeAgentFake:
    def __init__(self):
        self.calls = []

    def warmup_policy(self, batch_size):
        self.calls.append(("policy", batch_size))

    def warmup_train(self, imbalanced_reward_weights=None):
        self.calls.append(("train", imbalanced_reward_weights))

    def warmup_report(self):
        self.calls.append(("report",))


class TrainingSpeedFlagTest(unittest.TestCase):
    def setUp(self):
        self._old_env = {
            "LEDWM_FAST_TRAIN_METRICS": os.environ.get("LEDWM_FAST_TRAIN_METRICS"),
            "LEDWM_FAST_OPTIMIZER_METRICS": os.environ.get(
                "LEDWM_FAST_OPTIMIZER_METRICS"
            ),
            "LEDWM_SKIP_TRAIN_OUTS": os.environ.get("LEDWM_SKIP_TRAIN_OUTS"),
            "LEDWM_SKIP_ADAM_METRICS": os.environ.get("LEDWM_SKIP_ADAM_METRICS"),
        }
        os.environ.pop("LEDWM_FAST_TRAIN_METRICS", None)
        os.environ.pop("LEDWM_FAST_OPTIMIZER_METRICS", None)
        os.environ.pop("LEDWM_SKIP_TRAIN_OUTS", None)
        os.environ.pop("LEDWM_SKIP_ADAM_METRICS", None)

    def tearDown(self):
        for key, value in self._old_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    def test_train_outs_flag_skips_previous_result_and_submit(self):
        os.environ["LEDWM_SKIP_TRAIN_OUTS"] = "1"
        agent = AgentFake()
        previous = PromiseFake()
        agent.outs_promise = previous

        result = JAXAgent._process_train_outs(agent, {"post": 1})

        self.assertTrue(skip_train_outs_enabled())
        self.assertEqual(result, {})
        self.assertFalse(previous.result_called)
        self.assertEqual(agent.outs_worker.submitted, [])
        self.assertIsNone(agent.outs_promise)

    def test_train_outs_default_schedules_conversion(self):
        agent = AgentFake()

        result = JAXAgent._process_train_outs(agent, {"post": 1})

        self.assertFalse(skip_train_outs_enabled())
        self.assertEqual(result, {})
        self.assertEqual(len(agent.outs_worker.submitted), 1)
        self.assertIsNotNone(agent.outs_promise)

    def test_capacity_probe_includes_enabled_train_report(self):
        agent = CapacityProbeAgentFake()
        args = SimpleNamespace(
            actor_batch=200,
            report=SimpleNamespace(train=True),
        )
        weights = {-1.0: 1.0, 1.0: 1.0}

        warmup_agent_for_capacity_probe(agent, args, weights)

        self.assertEqual(
            agent.calls,
            [("policy", 200), ("train", weights), ("report",)],
        )

    def test_capacity_probe_skips_disabled_train_report(self):
        agent = CapacityProbeAgentFake()
        args = SimpleNamespace(
            actor_batch=32,
            report=SimpleNamespace(train=False),
        )

        warmup_agent_for_capacity_probe(agent, args)

        self.assertEqual(agent.calls, [("policy", 32), ("train", None)])

    def test_capacity_probe_can_skip_constant_policy_path(self):
        agent = CapacityProbeAgentFake()
        args = SimpleNamespace(
            actor_batch=32,
            report=SimpleNamespace(train=True),
        )

        warmup_agent_for_capacity_probe(agent, args, include_policy=False)

        self.assertEqual(agent.calls, [("train", None), ("report",)])

    def test_report_warmup_uses_full_training_batch(self):
        agent = object.__new__(JAXAgent)
        agent.batch_size = 7
        agent.batch_length = 11
        agent._train_spaces = object()
        agent.train_devices = ("gpu0",)
        agent.config_jax = SimpleNamespace(opt_step=True)
        observed = {}

        def dummy_batch(spaces, dims):
            observed["spaces"] = spaces
            observed["dims"] = dims
            return {
                "token_ids": np.zeros(dims, dtype=np.int64),
                "value": np.zeros(dims, dtype=np.float32),
            }

        def convert_inps(data, devices):
            observed["devices"] = devices
            return data

        def report(data, step=None):
            observed["data"] = data
            observed["step"] = step
            return {"metric": np.zeros(())}

        agent._dummy_batch = dummy_batch
        agent._convert_inps = convert_inps
        agent._next_rngs = lambda devices: "rng"
        agent.report = report

        JAXAgent.warmup_report(agent)

        self.assertIs(observed["spaces"], agent._train_spaces)
        self.assertEqual(observed["dims"], (7, 11))
        self.assertEqual(observed["devices"], ("gpu0",))
        self.assertEqual(observed["data"]["token_ids"].dtype, np.int32)
        self.assertEqual(observed["data"]["reward_indicator"].shape, (7,))
        self.assertEqual(observed["data"]["sample_id"].shape, (7, 16))
        self.assertEqual(observed["data"]["rng"], "rng")
        self.assertEqual(observed["step"], 0)

    def test_adam_metrics_flag(self):
        self.assertFalse(skip_adam_metrics_enabled())

        os.environ["LEDWM_SKIP_ADAM_METRICS"] = "1"

        self.assertTrue(skip_adam_metrics_enabled())

    def test_optimizer_metrics_flag(self):
        self.assertFalse(fast_optimizer_metrics_enabled())

        os.environ["LEDWM_FAST_OPTIMIZER_METRICS"] = "1"

        self.assertTrue(fast_optimizer_metrics_enabled())

    def test_task_config_enables_speed_flags_for_direct_python_launch(self):
        config = SimpleNamespace(
            run=SimpleNamespace(
                fast_train_metrics=True,
                fast_optimizer_metrics=True,
                skip_adam_metrics=True,
                skip_train_outs=True,
            )
        )

        configure_training_speed_flags(config)

        self.assertEqual(os.environ["LEDWM_FAST_TRAIN_METRICS"], "1")
        self.assertEqual(os.environ["LEDWM_FAST_OPTIMIZER_METRICS"], "1")
        self.assertEqual(os.environ["LEDWM_SKIP_ADAM_METRICS"], "1")
        self.assertEqual(os.environ["LEDWM_SKIP_TRAIN_OUTS"], "1")

    def test_explicit_environment_override_wins_over_task_config(self):
        os.environ["LEDWM_FAST_TRAIN_METRICS"] = "0"
        config = SimpleNamespace(
            run=SimpleNamespace(
                fast_train_metrics=True,
                fast_optimizer_metrics=True,
                skip_adam_metrics=False,
                skip_train_outs=False,
            )
        )

        configure_training_speed_flags(config)

        self.assertEqual(os.environ["LEDWM_FAST_TRAIN_METRICS"], "0")
