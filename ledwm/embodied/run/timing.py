import os
import threading
import time
from collections import OrderedDict
from contextlib import contextmanager

from ledwm.logging_setup import logger as event_logger


TRUE_VALUES = {"1", "true", "yes", "on"}


def progress_safe_print(message, flush=False):
    """Write a timestamped startup status line through Loguru."""
    event_logger.info(message)
    if flush:
        event_logger.complete()


@contextmanager
def long_operation_progress(
    label, detail="", every=15.0, print_fn=progress_safe_print
):
    """Print heartbeats while a startup operation blocks in JAX or replay."""
    started = time.monotonic()
    stopped = threading.Event()

    def detail_suffix():
        current = detail() if callable(detail) else detail
        return f" | detail={current}" if current else ""

    suffix = detail_suffix()
    print_fn(f"startup.operation | name={label} | state=started{suffix}", flush=True)

    def heartbeat():
        while not stopped.wait(every):
            elapsed = time.monotonic() - started
            print_fn(
                f"startup.operation | name={label} | state=running | "
                f"elapsed={elapsed:.0f}s{detail_suffix()}",
                flush=True,
            )

    thread = threading.Thread(
        target=heartbeat,
        name=f"startup-progress-{label}",
        daemon=True,
    )
    thread.start()
    try:
        yield
    except BaseException:
        elapsed = time.monotonic() - started
        print_fn(
            f"startup.operation | name={label} | state=failed | "
            f"elapsed={elapsed:.1f}s",
            flush=True,
        )
        raise
    else:
        elapsed = time.monotonic() - started
        print_fn(
            f"startup.operation | name={label} | state=completed | "
            f"elapsed={elapsed:.1f}s",
            flush=True,
        )
    finally:
        stopped.set()
        thread.join()


def has_tree_leaves(value):
    if value is None:
        return False
    if isinstance(value, dict):
        return any(has_tree_leaves(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(has_tree_leaves(item) for item in value)
    return True


def _bool_from_value(value):
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in TRUE_VALUES


def timing_enabled(args=None, attr="actor_timing", env="LEDWM_ACTOR_TIMING"):
    value = os.environ.get(env)
    if value is not None:
        return _bool_from_value(value)
    return _bool_from_value(getattr(args, attr, False))


def timing_every(
    args=None,
    attr="actor_timing_every",
    env="LEDWM_ACTOR_TIMING_EVERY",
    default=30.0,
):
    value = os.environ.get(env, getattr(args, attr, default))
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


class PeriodicTiming:
    def __init__(
        self, name, every=30.0, now=time.time, print_fn=progress_safe_print
    ):
        self.name = name
        self.every = float(every)
        self.now = now
        self.print_fn = print_fn
        self.last = self.now()
        self.events = 0
        self.totals = OrderedDict()
        self.counters = OrderedDict()

    def event(self, count=1):
        self.events += int(count)

    def record(self, key, seconds):
        self.totals[key] = self.totals.get(key, 0.0) + float(seconds)

    def increment(self, key, amount=1):
        self.counters[key] = self.counters.get(key, 0.0) + float(amount)

    def maybe_report(self):
        now = self.now()
        if now - self.last < self.every:
            return None
        return self.report(now)

    def report(self, now=None):
        now = self.now() if now is None else now
        metrics = self.metrics(now)
        parts = [
            f"count={int(metrics['count'])}",
            f"window={metrics['window']:.1f}s",
        ]

        for key in self.totals:
            parts.append(f"{key}={metrics[key]:.4f}s")
            parts.append(f"{key}_total={metrics[f'{key}_total']:.2f}s")

        for key in self.counters:
            parts.append(f"{key}={metrics[key]:.0f}")
            parts.append(f"{key}_rate={metrics[f'{key}_rate']:.2f}/s")

        event = self.name.replace(":", ".") + ".timing"
        self.print_fn(f"{event} | " + " | ".join(parts), flush=True)
        self.reset(now)
        return metrics

    def metrics(self, now=None):
        now = self.now() if now is None else now
        window = max(now - self.last, 1e-9)
        count = max(self.events, 1)
        result = OrderedDict([("count", self.events), ("window", window)])

        for key, total in self.totals.items():
            result[key] = total / count
            result[f"{key}_total"] = total

        for key, value in self.counters.items():
            result[key] = value
            result[f"{key}_rate"] = value / window

        return result

    def reset(self, now=None):
        self.last = self.now() if now is None else now
        self.events = 0
        self.totals.clear()
        self.counters.clear()


def make_timing(
    name,
    args=None,
    attr="actor_timing",
    env="LEDWM_ACTOR_TIMING",
    every_attr="actor_timing_every",
    every_env="LEDWM_ACTOR_TIMING_EVERY",
    default=30.0,
):
    if not timing_enabled(args, attr=attr, env=env):
        return None
    return PeriodicTiming(
        name,
        every=timing_every(args, attr=every_attr, env=every_env, default=default),
    )


def record_timing_breakdown(timing, prefix, breakdown):
    if not breakdown:
        return
    for key, seconds in breakdown.items():
        if seconds:
            timing.record(f"{prefix}_{key}", seconds)


def maybe_report_to_logger(timing, logger, prefix):
    metrics = timing.maybe_report()
    if metrics is None:
        return False
    logger.add(metrics, prefix=prefix)
    return True
