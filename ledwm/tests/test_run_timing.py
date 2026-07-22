from ledwm.embodied.run.timing import (
    PeriodicTiming,
    long_operation_progress,
    maybe_report_to_logger,
    progress_safe_print,
    record_timing_breakdown,
    timing_enabled,
)


def test_long_operation_progress_reports_start_heartbeat_and_completion():
    import time

    messages = []

    with long_operation_progress(
        "first JIT",
        "batch=100",
        every=0.01,
        print_fn=lambda message, flush=False: messages.append(message),
    ):
        time.sleep(0.025)

    assert messages[0] == (
        "startup.operation | name=first JIT | state=started | detail=batch=100"
    )
    assert any("state=running" in message for message in messages)
    assert messages[-1].startswith(
        "startup.operation | name=first JIT | state=completed | elapsed="
    )


def test_long_operation_progress_refreshes_callable_detail():
    import time

    messages = []
    progress = iter(["replay=0/64", "replay=32/64", "replay=64/64"])

    with long_operation_progress(
        "first batch",
        lambda: next(progress),
        every=0.01,
        print_fn=lambda message, flush=False: messages.append(message),
    ):
        time.sleep(0.025)

    assert messages[0] == (
        "startup.operation | name=first batch | state=started | detail=replay=0/64"
    )
    assert any("replay=32/64" in message for message in messages)


def test_progress_safe_print_uses_loguru(monkeypatch):
    calls = []

    class EventLogger:
        def info(self, message):
            calls.append(("info", message))

        def complete(self):
            calls.append(("complete", None))

    monkeypatch.setattr(
        "ledwm.embodied.run.timing.event_logger", EventLogger()
    )

    progress_safe_print("startup.operation | state=running", flush=True)

    assert calls == [
        ("info", "startup.operation | state=running"),
        ("complete", None),
    ]


def test_timing_enabled_accepts_env_var(monkeypatch):
    monkeypatch.setenv("LEDWM_ACTOR_TIMING", "1")

    assert timing_enabled()


def test_timing_enabled_rejects_empty_env_var(monkeypatch):
    monkeypatch.setenv("LEDWM_ACTOR_TIMING", "")

    assert not timing_enabled()


def test_env_timing_does_not_follow_actor_timing_by_default(monkeypatch):
    class Args:
        actor_timing = True
        env_timing = False

    monkeypatch.delenv("LEDWM_ENV_TIMING", raising=False)

    assert not timing_enabled(Args, attr="env_timing", env="LEDWM_ENV_TIMING")


def test_env_timing_can_be_enabled_explicitly(monkeypatch):
    class Args:
        actor_timing = False
        env_timing = True

    monkeypatch.delenv("LEDWM_ENV_TIMING", raising=False)

    assert timing_enabled(Args, attr="env_timing", env="LEDWM_ENV_TIMING")


def test_periodic_timing_reports_averages_and_rates():
    times = iter([0.0, 10.0])
    messages = []
    timing = PeriodicTiming(
        "actor",
        every=0,
        now=lambda: next(times),
        print_fn=lambda message, flush=False: messages.append(message),
    )

    timing.event()
    timing.record("policy", 0.2)
    timing.record("train_replay_add", 0.3)
    timing.increment("env_steps", 32)

    assert timing.maybe_report()

    assert "actor.timing |" in messages[0]
    assert "policy=0.2000s" in messages[0]
    assert "train_replay_add=0.3000s" in messages[0]
    assert "env_steps_rate=3.20/s" in messages[0]


def test_periodic_timing_returns_logger_scalars(capsys):
    times = iter([0.0, 10.0])
    timing = PeriodicTiming("actor", every=0, now=lambda: next(times))

    timing.event()
    timing.record("policy", 0.2)
    timing.increment("env_steps", 32)

    metrics = timing.maybe_report()

    assert metrics["policy"] == 0.2
    assert metrics["policy_total"] == 0.2
    assert metrics["env_steps"] == 32
    assert metrics["env_steps_rate"] == 3.2
    assert metrics["count"] == 1
    assert metrics["window"] == 10.0


def test_maybe_report_to_logger_adds_prefixed_metrics(capsys):
    class Logger:
        def __init__(self):
            self.calls = []

        def add(self, metrics, prefix=None):
            self.calls.append((metrics, prefix))

    times = iter([0.0, 10.0])
    timing = PeriodicTiming("actor", every=0, now=lambda: next(times))
    logger = Logger()
    timing.event()
    timing.record("policy", 0.2)

    assert maybe_report_to_logger(timing, logger, "actor_timing")

    metrics, prefix = logger.calls[0]
    assert prefix == "actor_timing"
    assert metrics["policy"] == 0.2


def test_record_timing_breakdown_records_prefixed_fields(capsys):
    times = iter([0.0, 10.0])
    timing = PeriodicTiming("actor", every=0, now=lambda: next(times))

    timing.event()
    timing.record("policy", 0.5)
    record_timing_breakdown(
        timing,
        "policy",
        {
            "preprocess": 0.1,
            "obs_device_put": 0.2,
            "zero": 0.0,
        },
    )

    metrics = timing.maybe_report()

    assert metrics["policy_preprocess"] == 0.1
    assert metrics["policy_obs_device_put"] == 0.2
    assert "policy_zero" not in metrics
