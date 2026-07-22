import concurrent.futures
from collections import deque

import zmq

from ledwm.embodied.core.distr import Server


class EmptySocket:
    def recv_multipart(self, flags):
        raise zmq.Again

    def send_multipart(self, message):
        raise AssertionError(f"unexpected send: {message}")


class RecordingExecutor:
    def __init__(self):
        self.calls = []

    def submit(self, function, addrs, inputs):
        self.calls.append((function, addrs, inputs))
        return concurrent.futures.Future()


def make_server(batch=2, max_inflight=3):
    server = object.__new__(Server)
    server.socket = EmptySocket()
    server.function = lambda inputs, addrs: None
    server.batch = batch
    server.workers = RecordingExecutor()
    server.max_inflight = max_inflight
    server.promises = deque()
    server.inputs = deque()
    server.requests = {}
    server.outputs = {}
    server.error = None
    return server


def test_step_dispatches_all_full_batches_up_to_inflight_limit():
    server = make_server(batch=2, max_inflight=3)
    server.inputs.extend((bytes([index]), b"payload") for index in range(8))

    server._step()

    assert len(server.workers.calls) == 3
    assert len(server.promises) == 3
    assert len(server.inputs) == 2


def test_step_reaps_completed_batches_out_of_order_before_dispatch():
    server = make_server(batch=2, max_inflight=2)
    slow = concurrent.futures.Future()
    finished = concurrent.futures.Future()
    finished.set_result(None)
    server.promises.extend((slow, finished))
    server.inputs.extend(((b"a", b"one"), (b"b", b"two")))

    server._step()

    assert list(server.promises)[0] is slow
    assert len(server.promises) == 2
    assert len(server.workers.calls) == 1
    assert not server.inputs
