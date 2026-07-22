import unittest
import threading
from collections import defaultdict
from unittest import mock

import numpy as np

from ledwm.embodied.replay.generic import (
    GenericReplay,
    REPLAY_SAMPLE_COUNT_KEY,
    REPLAY_SAMPLE_PRIORITY_KEY,
    consume_replay_sample_stats,
)
from ledwm.jaxagent import JAXAgent


class ReplayLoggingTest(unittest.TestCase):
    class _Sampler:
        key2priority = {"a": 2.0, "b": 4.0}

        def __len__(self):
            return 2

        def sample_batch(self, size):
            assert size == 2
            return ["a", "b"]

    class _Limiter:
        @staticmethod
        def want_sample():
            return True, "ok"

    class _Progress:
        def __init__(self, **kwargs):
            self.n = kwargs["initial"]
            self.updates = []
            self.closed = False

        def update(self, amount):
            self.n += amount
            self.updates.append(amount)

        def close(self):
            self.closed = True

    @staticmethod
    def _progress_replay(size=1, minimum=4):
        replay = object.__new__(GenericReplay)
        replay.finetune_online = False
        replay.table = {index: None for index in range(size)}
        replay.table_pos = {}
        replay.min_size = minimum
        replay._fill_progress = None
        replay._fill_progress_done = False
        replay._fill_progress_lock = threading.Lock()
        return replay

    def test_live_inserts_use_one_progress_bar_until_replay_is_ready(self):
        replay = self._progress_replay()
        progress = self._Progress(
            total=4,
            initial=1,
            desc="replay.fill",
            unit="seq",
            dynamic_ncols=True,
            mininterval=0.25,
        )
        with mock.patch(
            "ledwm.embodied.replay.generic.tqdm", return_value=progress
        ) as tqdm:
            replay._update_fill_progress(load=False)
            replay.table[1] = None
            replay._update_fill_progress(load=False)
            replay.table.update({2: None, 3: None})
            replay._update_fill_progress(load=False)

        tqdm.assert_called_once_with(
            total=4,
            initial=1,
            desc="replay.fill",
            unit="seq",
            dynamic_ncols=True,
            mininterval=0.25,
        )
        self.assertEqual(progress.updates, [1, 2])
        self.assertTrue(progress.closed)
        self.assertTrue(replay._fill_progress_done)

    @mock.patch("ledwm.embodied.replay.generic.tqdm")
    def test_restore_does_not_open_a_second_progress_bar(self, tqdm):
        replay = self._progress_replay(size=4, minimum=4)

        replay._update_fill_progress(load=True)

        tqdm.assert_not_called()
        self.assertTrue(replay._fill_progress_done)

    def test_load_summary_formats_numpy_reward_counts_as_plain_numbers(self):
        replay = object.__new__(GenericReplay)
        replay.is_eval = False
        replay.finetune_online = False
        replay.table = {"sequence": None}
        replay.table_pos = {}
        replay.upsample_pos = False
        replay.counts_eps_reward = {np.float32(-1.0): np.int64(123)}
        replay.counts_stream_reward = {}

        with mock.patch("ledwm.embodied.replay.generic.cprint") as cprint:
            replay.print(event="load_done")

        cprint.assert_called_once_with(
            "replay.load_done | mode=train | buffer_size=1 | "
            "pos_eps_rate=0.000 | pos_stream_rate=0.000 | "
            "reward_counts={-1.0: 123}",
            None,
        )

    def test_unique_batch_carries_stats_for_the_consumed_batch(self):
        replay = object.__new__(GenericReplay)
        replay.is_eval = False
        replay.lock = threading.Lock()
        replay.sampler = self._Sampler()
        replay.limiter = self._Limiter()
        replay._materialized = {
            "a": {"value": np.asarray([1])},
            "b": {"value": np.asarray([2])},
        }
        replay.counts_stream2sample = defaultdict(int, {"a": 2, "b": 4})
        replay.batch_counts_sampled = []
        replay.batch_priorities = []
        replay.train_ratio = 100
        replay.metrics = {
            "samples": 0,
            "sample_wait_dur": 0,
            "sample_wait_count": 0,
        }

        with mock.patch.object(
            GenericReplay,
            "supports_unique_batch_sampling",
            new_callable=mock.PropertyMock,
            return_value=True,
        ):
            samples = replay.sample_batch(2)

        batch = {
            key: np.stack([sample[key] for sample in samples])
            for key in samples[0]
        }
        mean_count, mean_priority = consume_replay_sample_stats(batch)

        self.assertEqual(mean_count, 4.0)
        self.assertEqual(mean_priority, 3.0)
        self.assertNotIn(REPLAY_SAMPLE_COUNT_KEY, batch)
        self.assertNotIn(REPLAY_SAMPLE_PRIORITY_KEY, batch)
        np.testing.assert_array_equal(batch["value"], [[1], [2]])

    def test_replay_stats_stay_on_host_during_batch_device_put(self):
        agent = object.__new__(JAXAgent)
        agent.train_devices = ["gpu"]
        converted_keys = []

        def convert(inputs, devices):
            self.assertEqual(devices, ["gpu"])
            converted_keys.extend(inputs)
            return {key: f"device:{key}" for key in inputs}

        agent._convert_inps = convert
        agent._next_rngs = lambda devices: "device:rng"
        counts = np.asarray([2, 4])
        priorities = np.asarray([1.5, 2.5])
        batch = {
            "value": np.asarray([[1], [2]]),
            REPLAY_SAMPLE_COUNT_KEY: counts,
            REPLAY_SAMPLE_PRIORITY_KEY: priorities,
        }

        result = agent._postprocess_dataset_batch(batch)

        self.assertEqual(converted_keys, ["value"])
        self.assertEqual(result["value"], "device:value")
        self.assertEqual(result["rng"], "device:rng")
        self.assertIs(result[REPLAY_SAMPLE_COUNT_KEY], counts)
        self.assertIs(result[REPLAY_SAMPLE_PRIORITY_KEY], priorities)


if __name__ == "__main__":
    unittest.main()
