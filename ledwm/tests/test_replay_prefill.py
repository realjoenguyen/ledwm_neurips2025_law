import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

from ledwm.embodied.core.config import Config
from ledwm.embodied.replay import bundle as bundlelib
from ledwm.embodied.replay import chunk as chunklib
from ledwm.embodied.replay import replays
from ledwm.embodied.replay.generic import (
    GenericReplay,
    _canonical_reward_indicator,
)
from ledwm.embodied.replay.saver import Saver


class ReplayPrefillTest(unittest.TestCase):
    def test_latest_compatible_prefill(self):
        signature = {"version": 1, "task": "messenger_s1", "batch_length": 50}
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            compatible = root / "old" / "episodes"
            incompatible = root / "newer" / "episodes"
            current = root / "current" / "episodes"
            for directory in (compatible, incompatible, current):
                directory.mkdir(parents=True)
            replays._write_replay_manifest(compatible, signature)
            replays._write_replay_manifest(
                incompatible, {**signature, "batch_length": 150}
            )
            (compatible / "saved.npz").touch()
            (incompatible / "saved.npz").touch()

            result = replays._latest_compatible_prefill(current, signature)

            self.assertEqual(result, compatible.resolve())

    def test_resolve_auto_prefill_can_be_disabled(self):
        with tempfile.TemporaryDirectory() as tempdir:
            directory = Path(tempdir) / "current" / "episodes"
            config = SimpleNamespace(
                replay=SimpleNamespace(
                    resume=False,
                    flat={"prefill": ""},
                )
            )

            self.assertIsNone(
                replays._resolve_replay_prefill(config, directory, {"version": 1})
            )


class SaverFlushTest(unittest.TestCase):
    def test_chunk_save_logs_each_file_at_debug(self):
        with tempfile.TemporaryDirectory() as tempdir:
            chunk = chunklib.Chunk(1)
            chunk.append({"value": np.asarray(1)})

            with mock.patch.object(chunklib, "event_logger") as event_logger:
                filename = chunk.save(tempdir)

            event_logger.debug.assert_called_once_with(
                f"replay.chunk_saved | file={filename.name}"
            )

    def test_load_progress_is_weighted_by_steps(self):
        with tempfile.TemporaryDirectory() as tempdir:
            saver = Saver(tempdir, chunks=10)
            saver.add({"value": np.asarray(1)}, worker=7)
            saver.add({"value": np.asarray(2)}, worker=7)
            saver.save(wait=True)
            saver.workers.shutdown(wait=True)

            loaded = Saver(tempdir, chunks=10)
            with mock.patch("ledwm.embodied.replay.saver.tqdm") as tqdm:
                steps = list(loaded.load(capacity=10, batch_length=2))
            loaded.workers.shutdown(wait=True)

            self.assertEqual(len(steps), 2)
            tqdm.assert_called_once_with(
                total=2,
                desc="replay.load",
                unit="step",
                dynamic_ncols=True,
            )
            tqdm.return_value.update.assert_called_once_with(2)
            tqdm.return_value.close.assert_called_once_with()
            self.assertFalse(loaded.loading)

    def test_incremental_save_merges_into_one_bundle_without_duplicates(self):
        with tempfile.TemporaryDirectory() as tempdir:
            saver = Saver(tempdir, chunks=10)
            saver.add({"value": np.asarray(1)}, worker=7)
            saver.save(wait=True)
            saver.add({"value": np.asarray(2)}, worker=7)
            saver.save(wait=True)
            saver.workers.shutdown(wait=True)

            files = sorted(path.name for path in Path(tempdir).glob("*.npz"))
            self.assertEqual(files, [bundlelib.FILENAME])
            loaded = Saver(tempdir, chunks=10)
            steps = list(loaded.load(capacity=10, batch_length=2))
            loaded.workers.shutdown(wait=True)
            values = [int(step["value"]) for step, _ in steps]
            self.assertEqual(values, [1, 2])

    def test_loads_legacy_chunks(self):
        with tempfile.TemporaryDirectory() as tempdir:
            chunk = chunklib.Chunk(2)
            chunk.append({"value": np.asarray(3)})
            chunk.append({"value": np.asarray(4)})
            chunk.save(tempdir)

            saver = Saver(tempdir, chunks=10)
            steps = list(saver.load(capacity=10, batch_length=2))
            saver.workers.shutdown(wait=True)

            self.assertEqual([int(step["value"]) for step, _ in steps], [3, 4])

    def test_bundle_preserves_worker_streams(self):
        with tempfile.TemporaryDirectory() as tempdir:
            saver = Saver(tempdir, chunks=2)
            for index in range(3):
                saver.add({"value": np.asarray(index)}, worker=7)
                saver.add({"value": np.asarray(10 + index)}, worker=9)
            saver.save(wait=True)
            saver.workers.shutdown(wait=True)

            loaded = Saver(tempdir, chunks=2)
            streams = {}
            for step, stream in loaded.load(capacity=20, batch_length=2):
                streams.setdefault(stream, []).append(int(step["value"]))
            loaded.workers.shutdown(wait=True)

            self.assertEqual(sorted(streams.values()), [[0, 1, 2], [10, 11, 12]])

    def test_bundle_preserves_worker_across_saver_restart(self):
        with tempfile.TemporaryDirectory() as tempdir:
            first = Saver(tempdir, chunks=10)
            first.add({"value": np.asarray(1)}, worker=7)
            first.save(wait=True)
            first.workers.shutdown(wait=True)

            second = Saver(tempdir, chunks=10)
            second.add({"value": np.asarray(2)}, worker=7)
            second.save(wait=True)
            second.workers.shutdown(wait=True)

            loaded = Saver(tempdir, chunks=10)
            streams = {}
            for step, stream in loaded.load(capacity=10, batch_length=2):
                streams.setdefault(stream, []).append(int(step["value"]))
            loaded.workers.shutdown(wait=True)

            self.assertEqual(list(streams.values()), [[1, 2]])

    def test_bundle_is_trimmed_to_replay_capacity(self):
        with tempfile.TemporaryDirectory() as tempdir:
            saver = Saver(tempdir, chunks=2, capacity=1, batch_length=2)
            for index in range(4):
                saver.add({"value": np.asarray(index)}, worker=0)
            saver.save(wait=True)
            saver.workers.shutdown(wait=True)

            chunks = bundlelib.load(tempdir)
            values = [int(value) for chunk in chunks for value in chunk.data["value"]]
            self.assertEqual(values, [2, 3])

    def test_preload_credits_rate_limiter(self):
        config = Config(
            {
                "task": "messenger_s1",
                "dataset_exclude_keys": None,
                "run": {"script": "parallel_train", "overfit_eps": False},
                "replay": {
                    "imbalance": "none",
                    "imbalance_reward": "sum",
                    "is_first": False,
                    "resume": False,
                    "remove_oversample": False,
                },
            }
        )
        with tempfile.TemporaryDirectory() as tempdir:
            source = Path(tempdir) / "source"
            destination = Path(tempdir) / "destination"
            saver = Saver(source, chunks=10)
            for index in range(3):
                saver.add(
                    {
                        "entity_pos": np.ones((3, 3), np.int32),
                        "is_first": np.bool_(index == 0),
                        "is_last": np.bool_(index == 2),
                        "reward_indicator": np.float32(1),
                    },
                    worker=0,
                )
            saver.save(wait=True)
            saver.workers.shutdown(wait=True)

            replay = replays.Uniform(
                length=3,
                is_eval=False,
                capacity=10,
                directory=destination,
                min_size=1,
                samples_per_insert=1,
                tolerance=10,
                train_ratio=1,
                load_directories=[source],
                config=config,
            )

            self.assertEqual(len(replay), 1)
            self.assertEqual(replay.limiter.avail, replay.limiter.max_avail)
            self.assertAlmostEqual(replay.pos_stream_rate, 1, places=5)
            self.assertEqual(replay.pos_eps_rate, 1)

    def test_legacy_chunks_repack_sparse_positive_episodes(self):
        config = Config(
            {
                "task": "messenger_s1",
                "dataset_exclude_keys": None,
                "run": {
                    "script": "parallel_train",
                    "overfit_eps": False,
                    "debug": False,
                },
                "replay": {
                    "imbalance": "none",
                    "imbalance_reward": "sum",
                    "is_first": True,
                    "resume": False,
                    "remove_oversample": False,
                },
                "overfit_batch": False,
            }
        )

        def episode(indicator):
            return [
                {
                    "entity_pos": np.ones((3, 3), np.int32),
                    "is_first": np.bool_(index == 0),
                    "is_last": np.bool_(index == 2),
                    "reward": np.float32(indicator if index == 2 else 0),
                    "reward_indicator": np.float32(indicator),
                }
                for index in range(3)
            ]

        with tempfile.TemporaryDirectory() as tempdir:
            source = Path(tempdir) / "source"
            destination = Path(tempdir) / "destination"
            source.mkdir()
            chunks = []
            for _ in range(2):
                chunk = chunklib.Chunk(6)
                for step in episode(1) + episode(-1):
                    chunk.append(step)
                chunk.worker = None  # Simulate a version-1 bundle.
                chunks.append(chunk)
            bundlelib.save(source, chunks)

            replay = replays.Uniform(
                length=5,
                is_eval=False,
                capacity=20,
                directory=destination,
                min_size=1,
                samples_per_insert=1,
                tolerance=20,
                train_ratio=1,
                load_directories=[source],
                config=config,
            )

            self.assertGreater(replay.counts_stream_reward[np.float32(1)], 0)
            self.assertGreater(replay.counts_stream_reward[np.float32(-1)], 0)
            self.assertGreater(replay.pos_stream_rate, 0)
            replay.saver.workers.shutdown(wait=True)


class ReplayLoadFastPathTest(unittest.TestCase):
    class _LoadSaver:
        def __init__(self, steps):
            self.steps = steps

        def load(self, capacity, batch_length, debug=False):
            del capacity, batch_length, debug
            for step in self.steps:
                yield step, "source"

    @staticmethod
    def _episode(indicators):
        return [
            {
                "is_first": np.bool_(index == 0),
                "is_last": np.bool_(index == len(indicators) - 1),
                "reward_indicator": np.float32(indicator),
            }
            for index, indicator in enumerate(indicators)
        ]

    def test_fast_canonicalization_matches_previous_numpy_bucketing(self):
        values = np.linspace(-3, 3, 12001, dtype=np.float32)
        expected = np.asarray(
            [np.round(value, 2).astype(np.float32) for value in values]
        )
        actual = np.asarray(
            [_canonical_reward_indicator(value) for value in values]
        )

        np.testing.assert_array_equal(actual, expected)

    def test_episode_repack_reuses_one_canonical_indicator(self):
        replay = object.__new__(GenericReplay)
        replay.is_first = True
        saver = self._LoadSaver(self._episode([1.001, 0.999, 1.004]))

        loaded = list(replay._load_complete_episodes(saver, 10, 3))

        self.assertEqual(len(loaded), 3)
        self.assertEqual({worker for _, worker in loaded}, {("replay_load", 1.0)})
        self.assertTrue(
            all(
                step["reward_indicator"] == np.float32(1.0)
                for step, _ in loaded
            )
        )

    def test_episode_repack_still_rejects_mixed_indicators(self):
        replay = object.__new__(GenericReplay)
        replay.is_first = True
        saver = self._LoadSaver(self._episode([1, -1]))

        with self.assertRaisesRegex(ValueError, "mixed reward indicators"):
            list(replay._load_complete_episodes(saver, 10, 2))


if __name__ == "__main__":
    unittest.main()
