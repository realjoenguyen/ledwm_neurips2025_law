import unittest
from unittest import mock

import numpy as np

from ledwm.embodied.core.config import Config
from ledwm.embodied.replay.replays import Uniform


def make_replay():
    config = Config(
        {
            "task": "messenger_s1",
            "dataset_exclude_keys": None,
            "run": {
                "script": "parallel_train",
                "overfit_eps": False,
            },
            "replay": {
                "imbalance": "none",
                "imbalance_reward": "sum",
                "is_first": False,
                "resume": False,
                "remove_oversample": False,
            },
        }
    )
    replay = Uniform(
        length=3,
        is_eval=False,
        capacity=10,
        min_size=1,
        train_ratio=128,
        config=config,
    )
    for time_step in range(3):
        replay.add(
            {
                "entity_pos": np.ones((3, 3), np.int32),
                "is_first": np.bool_(time_step == 0),
                "is_last": np.bool_(time_step == 2),
                "value": np.asarray([time_step, time_step + 1], np.int32),
            },
            worker=0,
        )
    return replay


class ReplayMaterializationCacheTest(unittest.TestCase):
    def test_loaded_steps_skip_redundant_preprocessing(self):
        replay = make_replay()

        with mock.patch.object(
            replay, "preprocess", side_effect=AssertionError("unexpected preprocess")
        ):
            for time_step in range(3):
                replay.add(
                    {
                        "entity_pos": np.ones((3, 3), np.int32),
                        "is_first": np.bool_(time_step == 0),
                        "is_last": np.bool_(time_step == 2),
                        "value": np.asarray([time_step], np.int32),
                    },
                    worker=1,
                    load=True,
                )

        self.assertEqual(len(replay), 2)

    def test_sample_reuses_cached_arrays_without_sharing_the_mapping(self):
        replay = make_replay()

        first = replay.sample()
        second = replay.sample()

        self.assertIsNot(first, second)
        self.assertIs(first["value"], second["value"])
        np.testing.assert_array_equal(first["value"], [[0, 1], [1, 2], [2, 3]])
        self.assertFalse(first["value"].flags.writeable)
        with self.assertRaisesRegex(ValueError, "read-only"):
            first["value"][0, 0] = 99

    def test_removal_and_reset_drop_materialized_sequences(self):
        replay = make_replay()
        self.assertEqual(len(replay._materialized), 1)
        self.assertGreater(replay._materialized_nbytes, 0)

        replay._remove()
        self.assertEqual(replay._materialized, {})
        self.assertEqual(replay._materialized_nbytes, 0)

        replay = make_replay()
        replay.reset()
        self.assertEqual(replay._materialized, {})
        self.assertEqual(replay._materialized_nbytes, 0)

    def test_deferred_sample_eviction_is_noop_after_capacity_eviction(self):
        replay = make_replay()
        for time_step in range(3):
            replay.add(
                {
                    "entity_pos": np.ones((3, 3), np.int32),
                    "is_first": np.bool_(time_step == 0),
                    "is_last": np.bool_(time_step == 2),
                    "value": np.asarray([time_step + 10], np.int32),
                },
                worker=1,
            )

        key = next(iter(replay.table))
        reward_indicator = replay.table[key][0]["reward_indicator"]

        # Simulate the actor's over-capacity eviction winning the race against
        # the learner's deferred sample-limit eviction for the same key.
        replay._remove()
        limiter_size = replay.limiter.size
        stream_reward_counts = dict(replay.counts_stream_reward)
        episode_reward_counts = dict(replay.counts_eps_reward)

        replay._remove_key(
            key,
            reward_indicator,
            force=True,
            preserve_sample_count=True,
        )

        self.assertNotIn(key, replay.table)
        self.assertEqual(replay.limiter.size, limiter_size)
        self.assertEqual(dict(replay.counts_stream_reward), stream_reward_counts)
        self.assertEqual(dict(replay.counts_eps_reward), episode_reward_counts)


if __name__ == "__main__":
    unittest.main()
