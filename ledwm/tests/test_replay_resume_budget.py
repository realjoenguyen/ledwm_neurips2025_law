import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from ledwm.embodied.core.checkpoint import Checkpoint
from ledwm.embodied.core.config import Config
from ledwm.embodied.replay.Prioritized import PrioritizedReplay
from ledwm.embodied.replay.generic import REPLAY_STEP_ID_KEY, ReplayProgress
from ledwm.embodied.replay.replays import Uniform


def make_config(resume):
    return Config(
        {
            "task": "messenger_s1",
            "dataset_exclude_keys": None,
            "overfit_batch": False,
            "run": {
                "script": "parallel_train",
                "overfit_eps": False,
                "debug": False,
            },
            "replay": {
                "imbalance": "none",
                "imbalance_reward": "sum",
                "is_first": True,
                "resume": resume,
                "remove_oversample": False,
            },
        }
    )


def make_replay(directory, resume):
    return Uniform(
        length=3,
        is_eval=False,
        capacity=10,
        directory=directory,
        chunks=10,
        min_size=1,
        samples_per_insert=1,
        tolerance=10,
        train_ratio=3,
        config=make_config(resume),
    )


def add_episode(replay, worker=0, value_offset=0):
    for time_step in range(3):
        replay.add(
            {
                "entity_pos": np.ones((3, 3), np.int32),
                "is_first": np.bool_(time_step == 0),
                "is_last": np.bool_(time_step == 2),
                "reward": np.float32(1 if time_step == 2 else 0),
                "value": np.asarray(value_offset + time_step, np.int32),
            },
            worker=worker,
            reward_indicator=1,
        )


class ReplayResumeBudgetTest(unittest.TestCase):
    def test_resume_only_samples_remaining_budget(self):
        fake_jaxutils = types.ModuleType("ledwm.jaxutils")
        fake_jaxutils.get_task = lambda config: "s1"
        with (
            mock.patch.dict(sys.modules, {"ledwm.jaxutils": fake_jaxutils}),
            tempfile.TemporaryDirectory() as directory,
        ):
            replay = make_replay(directory, resume=False)
            add_episode(replay)
            original_key = next(iter(replay.table))

            first = replay.sample()
            self.assertNotIn(REPLAY_STEP_ID_KEY, first)
            self.assertEqual(replay.counts_stream2sample[original_key], 1)
            expected_avail = replay.limiter.avail
            checkpoint_path = Path(directory) / "checkpoint.ckpt"
            checkpoint = Checkpoint(checkpoint_path, parallel=False)
            checkpoint.replay_progress = ReplayProgress(replay)
            checkpoint.save()
            replay.save(wait=True)
            replay.saver.workers.shutdown(wait=True)

            resumed = make_replay(directory, resume=True)
            resumed_key = next(iter(resumed.table))
            self.assertEqual(resumed_key, original_key)
            resumed_checkpoint = Checkpoint(checkpoint_path, parallel=False)
            resumed_checkpoint.replay_progress = ReplayProgress(resumed)
            resumed_checkpoint.load()

            self.assertEqual(resumed.counts_stream2sample[resumed_key], 1)
            self.assertEqual(resumed.limiter.avail, expected_avail)

            resumed.sample()
            self.assertEqual(resumed.counts_stream2sample[resumed_key], 2)
            self.assertEqual(len(resumed), 1)
            resumed.sample()
            self.assertEqual(resumed.counts_stream2sample[resumed_key], 3)
            self.assertEqual(len(resumed), 0)

            exhausted_progress = ReplayProgress(resumed).save()
            resumed.saver.workers.shutdown(wait=True)

            resumed_again = make_replay(directory, resume=True)
            self.assertEqual(len(resumed_again), 1)
            ReplayProgress(resumed_again).load(exhausted_progress)
            self.assertEqual(len(resumed_again), 0)
            resumed_again.saver.workers.shutdown(wait=True)

    def test_prioritized_batch_respects_restored_budget(self):
        fake_jaxutils = types.ModuleType("ledwm.jaxutils")
        fake_jaxutils.get_task = lambda config: "s1"
        sampler_options = {
            "exponent": 1,
            "initial": 1,
            "branching": 2,
            "eps": 0.001,
            "alpha": 2,
            "beta": 0.5,
            "c": 1e5,
        }

        def create(directory, resume):
            return PrioritizedReplay(
                length=3,
                train_ratio=2,
                capacity=10,
                directory=directory,
                chunks=10,
                tolerance=10,
                min_size=2,
                samples_per_insert=1,
                config=make_config(resume),
                **sampler_options,
            )

        with (
            mock.patch.dict(sys.modules, {"ledwm.jaxutils": fake_jaxutils}),
            tempfile.TemporaryDirectory() as directory,
        ):
            replay = create(directory, resume=False)
            add_episode(replay, worker=0, value_offset=0)
            add_episode(replay, worker=1, value_offset=10)
            replay.sample_batch(2)
            progress = ReplayProgress(replay).save()
            replay.save(wait=True)
            replay.saver.workers.shutdown(wait=True)

            resumed = create(directory, resume=True)
            ReplayProgress(resumed).load(progress)
            self.assertEqual(sorted(resumed.num_samples_whole_dataset), [1, 1])
            resumed.sample_batch(2)
            self.assertEqual(len(resumed), 0)
            self.assertEqual(
                sorted(resumed.save_progress()["sample_counts"].values()),
                [2, 2],
            )
            resumed.saver.workers.shutdown(wait=True)


if __name__ == "__main__":
    unittest.main()
