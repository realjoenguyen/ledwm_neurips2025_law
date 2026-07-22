from ledwm.embodied.run.async_replay import AsyncReplayAdder


class FakeReplay:

    def __init__(self):
        self.calls = []

    def add(self, tran, **kwargs):
        self.calls.append((tran, kwargs))


def test_async_replay_adder_runs_adds_in_worker_thread():
    replay = FakeReplay()
    adder = AsyncReplayAdder(maxsize=2)
    try:
        adder.add(replay, {"value": 1}, worker="env-a")
        adder.drain()
    finally:
        adder.close()

    assert replay.calls == [({"value": 1}, {"worker": "env-a"})]
