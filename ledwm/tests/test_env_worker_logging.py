from types import SimpleNamespace

import pytest

from ledwm.embodied.core import distr
from ledwm.embodied.run import env_worker
from ledwm.embodied.run.env_worker import EpisodeLogThrottle


class CapturingLogger:
    def __init__(self):
        self.records = []

    def _record(self, level, message, *args):
        self.records.append((level, message.format(*args)))

    def info(self, message, *args):
        self._record("INFO", message, *args)

    def debug(self, message, *args):
        self._record("DEBUG", message, *args)

    def warning(self, message, *args):
        self._record("WARNING", message, *args)

    def error(self, message, *args):
        self._record("ERROR", message, *args)


def test_episode_log_throttle_logs_first_then_waits_for_interval():
    times = iter([100.0, 105.0, 130.0])
    throttle = EpisodeLogThrottle(every=30, now=lambda: next(times))

    assert throttle()
    assert not throttle()
    assert throttle()


def test_episode_log_throttle_zero_disables_logging():
    throttle = EpisodeLogThrottle(every=0, now=lambda: 100.0)

    assert not throttle()


def test_episode_log_throttle_negative_preserves_unthrottled_logging():
    throttle = EpisodeLogThrottle(every=-1, now=lambda: 100.0)

    assert throttle()
    assert throttle()


def test_parallel_env_lifecycle_uses_loguru(monkeypatch):
    logger = CapturingLogger()
    monkeypatch.setattr(env_worker, "event_logger", logger)
    monkeypatch.setattr(env_worker, "configure_logging", lambda: None)

    class Space:
        def sample(self):
            return 0

    class Env:
        act_space = {"action": Space()}

        def step(self, act):
            return {"reward": 0.0, "is_last": False}

    class Client:
        def __init__(self, *args, **kwargs):
            pass

        def __call__(self, obs):
            def receive():
                raise distr.RemoteError(
                    "Server responded with an error: RESOURCE_EXHAUSTED"
                )

            return receive

    monkeypatch.setattr(env_worker.distr, "Client", Client)
    args = SimpleNamespace(
        actor_host="localhost",
        actor_port=5442,
        ipv6=False,
        actor_timeout=1,
        max_reconnect=1,
        episode_log_every=-1,
        first_step=False,
        env_timing=False,
        env_timing_every=30,
    )

    with pytest.raises(SystemExit):
        env_worker.parallel_env(46, Env, args, {})

    assert ("DEBUG", "env.create | env_id=46") in logger.records
    assert (
        "ERROR",
        "env.shutdown | env_id=46 | reason=agent_error | "
        "error=Server responded with an error: RESOURCE_EXHAUSTED",
    ) in logger.records


def test_rpc_client_status_uses_loguru(monkeypatch):
    logger = CapturingLogger()
    monkeypatch.setattr(distr, "event_logger", logger)
    client = object.__new__(distr.Client)
    client.identity = 46

    client._print("Client connecting to tcp://localhost:5442")
    client._print("Reconnecting", color="red")

    assert logger.records == [
        ("DEBUG", "[46] Client connecting to tcp://localhost:5442"),
        ("WARNING", "[46] Reconnecting"),
    ]
