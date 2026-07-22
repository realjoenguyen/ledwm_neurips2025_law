import threading

import numpy as np

from ledwm.embodied.core.batcher import Batcher


class Queue:
    maxsize = 4

    def qsize(self):
        return 2


def test_batcher_timing_reports_input_pipeline_phases(capsys):
    batcher = Batcher(
        sources=[],
        workers=0,
        preprocessors={},
        timing=True,
        timing_name="test",
        timing_every=0,
    )

    batcher._record_batcher_timing(
        {
            "get": 0.1,
            "stack": 0.2,
            "postprocess": 0.3,
            "output_wait": 0.4,
        },
        Queue(),
    )

    captured = capsys.readouterr()
    assert "batcher.timing | name=test |" in captured.out
    assert "get=" in captured.out
    assert "stack=" in captured.out
    assert "postprocess=" in captured.out
    assert "output_wait=" in captured.out


class UniqueBatchSource:
    supports_unique_batch_sampling = True

    def __init__(self):
        self.calls = 0
        self.lock = threading.Lock()

    def dataset(self):
        raise AssertionError("unique batch mode must call sample_batch()")

    def sample_batch(self, size):
        with self.lock:
            batch_id = self.calls
            self.calls += 1
        return [
            {
                "batch_id": np.asarray(batch_id),
                "sample_id": np.asarray(index),
                "value": np.asarray([batch_id, index]),
            }
            for index in range(size)
        ]


def test_unique_batch_uses_one_whole_batch_handoff():
    source = UniqueBatchSource()
    batcher = Batcher(
        sources=[source.dataset] * 32,
        workers=8,
        preprocessors={},
    )

    try:
        first = next(batcher)
        second = next(batcher)
    finally:
        batcher._running = False

    assert len(batcher._queues) == 1
    np.testing.assert_array_equal(first["batch_id"], np.zeros(32))
    np.testing.assert_array_equal(first["sample_id"], np.arange(32))
    np.testing.assert_array_equal(second["batch_id"], np.ones(32))
    np.testing.assert_array_equal(second["sample_id"], np.arange(32))
