import sys
import time

from ledwm.startup import configure_numeric_threading
from ledwm.logging_setup import configure_logging, logger as event_logger

configure_numeric_threading()

import numpy as np

from ledwm.embodied.core import distr
from ledwm.embodied.run.action_rpc import expand_action_from_rpc
from ledwm.embodied.run.timing import make_timing


class EpisodeLogThrottle:
    def __init__(self, every, now=time.time):
        self.every = every
        self.now = now
        self.last = None

    def __call__(self):
        if self.every < 0:
            return True
        if self.every == 0:
            return False

        now = self.now()
        if self.last is None or now >= self.last + self.every:
            self.last = now
            return True
        return False


def get_env_address(val: int):
    return val.to_bytes(16, "big").hex()


def print_eps_info(env_id, length, score, env_addr_types):
    env_address = get_env_address(env_id)
    env_type = env_addr_types[env_address]

    if env_type == "train":
        event_logger.debug(
            "env.episode | env_id={} | mode=train | length={} | score={:.4f}",
            env_id,
            length,
            score,
        )

    elif env_type == "eval":
        event_logger.debug(
            "env.episode | env_id={} | mode=eval | length={} | score={:.4f}",
            env_id,
            length,
            score,
        )

    elif env_type == "test":
        event_logger.debug(
            "env.episode | env_id={} | mode=test | length={} | score={:.4f}",
            env_id,
            length,
            score,
        )

    else:
        raise ValueError(f"Unknown env type: {env_type}")


def parallel_env(env_id, make_env, args, env_addr_types, timer=None):
    # TODO: Optionally write NPZ episodes.
    configure_logging()
    assert env_id >= 0, env_id
    event_logger.debug("env.create | env_id={}", env_id)
    env = make_env()
    timer and timer.wrap("env", env, ["step"])  # type: ignore
    addr = f"{args.actor_host}:{args.actor_port}"

    actor = distr.Client(
        addr,
        env_id,
        args.ipv6,
        timeout=distr.resolve_actor_timeout(args),
        max_reconnect=args.max_reconnect,
    )
    done = True
    act = None
    start = time.time()
    score = length = count = 0
    should_log_episode = EpisodeLogThrottle(getattr(args, "episode_log_every", -1))
    env_timing = make_timing(
        f"env:{env_id}",
        args,
        attr="env_timing",
        env="LEDWM_ENV_TIMING",
        every_attr="env_timing_every",
        every_env="LEDWM_ENV_TIMING_EVERY",
    )

    while True:
        loop_start = time.time()
        if done:
            act = {k: v.sample() for k, v in env.act_space.items()}
            act["reset"] = True
            score, length = 0, 0

        step_start = time.time()
        obs = env.step(act)
        env_step_time = time.time() - step_start

        obs = {k: np.asarray(v) for k, v in obs.items()}
        score += obs["reward"]
        length += 1  # reset counts as a step in here. SO length == 1 means reset

        done = obs["is_last"]
        if done:
            if should_log_episode():
                print_eps_info(env_id, length, score, env_addr_types)
            if args.first_step:
                assert length == 1, length

        rpc_send_start = time.time()
        promise = actor(obs)
        rpc_send_time = time.time() - rpc_send_start
        action_wait_time = 0.0
        try:
            action_wait_start = time.time()
            act = promise()
            action_wait_time = time.time() - action_wait_start
            act = {k: v for k, v in act.items() if not k.startswith("log_")}
            if "action" in env.act_space:
                act = expand_action_from_rpc(act, env.act_space["action"])

        except distr.ReconnectError:
            action_wait_time = time.time() - action_wait_start
            event_logger.warning(
                "env.reconnect | env_id={} | action=restart_episode", env_id
            )
            done = True

        except distr.RemoteError as e:
            action_wait_time = time.time() - action_wait_start
            event_logger.error(
                "env.shutdown | env_id={} | reason=agent_error | error={}",
                env_id,
                e,
            )
            sys.exit(0)

        if env_timing is not None:
            env_timing.event()
            env_timing.record("env_step", env_step_time)
            env_timing.record("rpc_send", rpc_send_time)
            env_timing.record("action_wait", action_wait_time)
            env_timing.record("loop", time.time() - loop_start)
            env_timing.increment("env_steps")
            env_timing.maybe_report()

        count += 1
        now = time.time()
        if now - start >= 60:
            fps = count / (now - start)
            event_logger.debug(
                "env.throughput | env_id={} | fps={:.1f}", env_id, fps
            )
            start = now
            count = 0
