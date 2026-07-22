import types
import os
import pathlib
import sys
import unittest

import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "messenger-emma"))

from ledwm.embodied.run.parallel import (
    _replay_startup_detail,
    _save_best_eval_if_improved,
    _train_reward_weights,
    _wait_for_first_agent_update,
    _warn_replay_starvation,
    parallel_learner,
)
from ledwm.embodied.core.counter import Counter


class StopAfterFirstTrain(Exception):
    pass


class CounterFake:
    def __init__(self, value=0):
        self.value = value

    def __int__(self):
        return self.value

    def increment(self, amount=1):
        self.value += amount


class DatasetFake:
    def __init__(self):
        self.next_count = 0

    def __iter__(self):
        return self

    def __next__(self):
        batch_id = self.next_count
        self.next_count += 1
        return {
            "is_first": np.zeros((1,), dtype=bool),
            "obs": np.full((1,), batch_id, dtype=np.int32),
        }


class AgentFake:
    def __init__(self, dataset):
        self.dataset_iter = dataset
        self.trained_batch_ids = []

    def dataset(self, _source):
        return iter(self.dataset_iter)

    def train(self, batch, **_kwargs):
        self.trained_batch_ids.append(int(batch["obs"][0]))
        raise StopAfterFirstTrain


class ReplayFake:
    dataset = object()
    batch_counts_sampled = []
    batch_priorities = []
    imbalanced_rewards = []
    limiter = types.SimpleNamespace(minimum=64)

    def __len__(self):
        return 32

    def clear_after_batch(self):
        pass

    def save(self):
        pass


class LoggerFake:
    def __init__(self):
        self.warnings = []

    def warning(self, message, *args):
        self.warnings.append(message.format(*args))


class BestEvalCheckpointFake:
    def __init__(self, best=0.0):
        self.best_eval_win_rate = Counter(best)
        self.best_eval_agent = None
        self.save_count = 0

    def save(self):
        self.save_count += 1


class ParallelLearnerWarmupTest(unittest.TestCase):
    def test_best_eval_checkpoint_preserves_fractional_comparison(self):
        checkpoint = BestEvalCheckpointFake(best=0.262)
        best_counter = checkpoint.best_eval_win_rate
        agent = object()

        self.assertFalse(
            _save_best_eval_if_improved(checkpoint, agent, np.float64(0.254))
        )
        self.assertEqual(checkpoint.save_count, 0)
        self.assertIs(checkpoint.best_eval_win_rate, best_counter)
        self.assertEqual(checkpoint.best_eval_win_rate.value, 0.262)

        self.assertTrue(
            _save_best_eval_if_improved(checkpoint, agent, np.float64(0.275))
        )
        self.assertEqual(checkpoint.save_count, 1)
        self.assertIs(checkpoint.best_eval_win_rate, best_counter)
        self.assertEqual(checkpoint.best_eval_win_rate.value, 0.275)
        self.assertIs(checkpoint.best_eval_agent, agent)

        self.assertFalse(_save_best_eval_if_improved(checkpoint, agent, 0.275))
        self.assertEqual(checkpoint.save_count, 1)

    def test_replay_startup_detail_distinguishes_data_wait_from_jit(self):
        self.assertEqual(
            _replay_startup_detail(ReplayFake()),
            "replay sequences=32/64; waiting for data, learner JIT has not started",
        )

    def test_eval_waits_for_first_completed_learner_update(self):
        agent = types.SimpleNamespace(updates=CounterFake())
        sleeps = []

        def finish_first_update(seconds):
            sleeps.append(seconds)
            agent.updates.increment()

        _wait_for_first_agent_update(agent, sleep=finish_first_update)

        self.assertEqual(sleeps, [0.25])

    def test_replay_starvation_warns_once_until_availability_recovers(self):
        logger = LoggerFake()
        starved = False

        for avail in (0.01, 0.0, -0.01, 0.25, 0.0):
            starved = _warn_replay_starvation(avail, starved, 42, log=logger)

        self.assertEqual(
            logger.warnings,
            [
                "replay.starved | avail=0.00% | opt_step=42 | "
                "state=waiting_for_insert_credit",
                "replay.starved | avail=0.00% | opt_step=42 | "
                "state=waiting_for_insert_credit",
            ],
        )

    def test_parallel_learner_trains_on_first_sampled_batch(self):
        dataset = DatasetFake()
        agent = AgentFake(dataset)
        args = types.SimpleNamespace(
            log_every=300,
            save_every=900,
            opt_step=False,
            report=types.SimpleNamespace(train=False),
        )
        config = types.SimpleNamespace(
            replay=types.SimpleNamespace(imbalance="none"),
            run=types.SimpleNamespace(debug=True),
        )

        with self.assertRaises(StopAfterFirstTrain):
            parallel_learner(
                CounterFake(),
                CounterFake(),
                CounterFake(),
                agent,
                ReplayFake(),
                logger=object(),
                checkpoint=object(),
                args=args,
                config=config,
            )

        self.assertEqual(agent.trained_batch_ids, [0])
        self.assertEqual(dataset.next_count, 1)

    def test_s1_uses_a_stable_reward_weight_signature(self):
        replay = ReplayFake()
        replay.reward_balanced_weight = lambda key: {-1.0: 1.0, 1.0: 4.0}[key]
        config = types.SimpleNamespace(
            task="s1",
            replay=types.SimpleNamespace(imbalance="balanced_weight"),
        )

        weights = _train_reward_weights(replay, config)

        self.assertEqual(list(weights.items()), [(-1.0, 1.0), (1.0, 4.0)])

    def test_s2_uses_a_stable_reward_weight_signature(self):
        replay = ReplayFake()
        replay.reward_balanced_weight = lambda key: {
            -1.0: 1.0,
            -0.5: 2.0,
            1.5: 8.0,
        }[key]
        config = types.SimpleNamespace(
            task="messenger_s2",
            replay=types.SimpleNamespace(imbalance="balanced_weight"),
        )

        weights = _train_reward_weights(replay, config)

        self.assertEqual(
            list(weights.items()),
            [(-1.0, 1.0), (-0.5, 2.0), (1.5, 8.0)],
        )

    def test_s3_uses_a_stable_reward_weight_signature(self):
        replay = ReplayFake()
        replay.reward_balanced_weight = lambda key: {
            -2.0: 1.0,
            -1.5: 2.0,
            -1.0: 3.0,
            -0.5: 4.0,
            1.5: 5.0,
        }[key]
        config = types.SimpleNamespace(
            task="messenger_s3",
            replay=types.SimpleNamespace(imbalance="balanced_weight"),
        )

        weights = _train_reward_weights(replay, config)

        self.assertEqual(
            list(weights.items()),
            [(-2.0, 1.0), (-1.5, 2.0), (-1.0, 3.0), (-0.5, 4.0), (1.5, 5.0)],
        )

    def test_lwm_tasks_use_a_stable_reward_weight_signature(self):
        for task in ("lwm_easy", "lwm_medium", "lwm_hard"):
            with self.subTest(task=task):
                replay = ReplayFake()
                replay.reward_balanced_weight = lambda key: {
                    -1.0: 1.0,
                    -0.5: 2.0,
                    1.5: 8.0,
                }[key]
                config = types.SimpleNamespace(
                    task=task,
                    replay=types.SimpleNamespace(imbalance="balanced_weight"),
                )

                weights = _train_reward_weights(replay, config)

                self.assertEqual(
                    list(weights.items()),
                    [(-1.0, 1.0), (-0.5, 2.0), (1.5, 8.0)],
                )
