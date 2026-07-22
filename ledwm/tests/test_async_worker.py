import pytest

from ledwm.embodied.run.async_worker import AsyncWorker


def test_async_worker_runs_submitted_calls_in_order():
    calls = []
    worker = AsyncWorker("test", maxsize=2)
    try:
        worker.submit(calls.append, 1)
        worker.submit(calls.append, 2)
        worker.drain()
    finally:
        worker.close()

    assert calls == [1, 2]


def test_async_worker_surfaces_worker_failures():
    worker = AsyncWorker("test", maxsize=2)

    def fail():
        raise ValueError("boom")

    try:
        worker.submit(fail)
        worker.drain()
    except RuntimeError as exc:
        assert "test failed" in str(exc)
        assert isinstance(exc.__cause__, ValueError)
    else:
        pytest.fail("expected worker failure")
    finally:
        worker.close(ignore_errors=True)
