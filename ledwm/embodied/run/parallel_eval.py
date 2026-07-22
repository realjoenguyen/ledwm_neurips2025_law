from multiprocessing import Lock, Value
import re
import sys
import time
from collections import defaultdict

try:
    from ledwm.startup import configure_tensorflow_cpp_warnings
except ModuleNotFoundError:
    from startup import configure_tensorflow_cpp_warnings

from ledwm.logging_setup import configure_logging, logger as event_logger

configure_tensorflow_cpp_warnings()

from tensorflow.keras.mixed_precision import set_global_policy  # type: ignore

# tying and TYPE_CHECKING are used to avoid circular imports.
from typing import TYPE_CHECKING, Dict

from termcolor import cprint
from ledwm.embodied.core.checkpoint import Checkpoint
from ledwm.embodied.core.counter import Counter
from ledwm.embodied.core.distr import MAX_RECONNECT_LOAD

from ledwm.embodied.run.eval_only import done_cal_win_rate


if TYPE_CHECKING:
    from ledwm.embodied.core.logger import Logger
    from ledwm.embodied.replay.replays import Uniform
    from ledwm.jaxagent import JAXAgent
    from ledwm.agent import Agent

from ledwm.embodied.run.parallel import dummy_data
import numpy as np

from ledwm.embodied.core.path import Path
from ledwm.embodied.core.timer import Timer
from ledwm.embodied.core.usage import Usage
from ledwm.embodied.core.when import Clock
from ledwm.embodied.core.metrics import Metrics
from ledwm.embodied.core.basics import treemap, convert
from ledwm.embodied.core import distr

start_parallel = 0
# share variable among multiple processes
total_step_time = Value("d", 0.0)
lock = Lock()
win_rates = []
WIN_REWARDS = {
    "s1": 1,
    "s2": 1.5,
    "s3": 1.5,
    "lwm_easy": 1,
    "lwm_medium": 1.5,
    "lwm_hard": 1.5,
    "messenger_s1": 1,
    "messenger_s2": 1.5,
    "messenger_s3": 1.5,
}


def parallel_eval(
    agent: "JAXAgent", logger: "Logger", make_env, num_envs: int, args, config
):
    cprint("eval.start", "green")
    real_env_step = embodied.Counter()
    timer = Timer()
    timer.wrap("agent", agent, ["policy", "train", "report", "save"])
    timer.wrap("logger", logger, ["write"])
    usage = embodied.Usage(args.trace_malloc)
    workers = []
    event_logger.info("run.workers_start | mode=eval")
    global start_parallel
    start_parallel = time.time()
    eps_cnt = Counter()

    if num_envs == 1:
        workers.append(distr.Thread(parallel_env_eval, 0, make_env, args, timer))
    else:
        for i in range(num_envs):
            worker = distr.Process(parallel_env_eval, i, make_env, args)
            worker.start()
            workers.append(worker)
        usage.processes("envs", workers)  # envs_count

    workers.append(
        distr.Thread(
            parallel_actor_eval,
            eps_cnt,
            logger.step,
            real_env_step,
            agent,
            logger,
            timer,
            args,
            config,
        )
    )

    distr.run(workers)


def parallel_actor_eval(
    eps_cnt: "Counter",
    step: "Counter",
    real_env_step: "Counter",
    agent: "JAXAgent",
    logger: "Logger",
    timer,
    args,
    config,
):
    metrics = Metrics()
    scalars = defaultdict(lambda: defaultdict(list))
    videos = defaultdict(lambda: defaultdict(list))
    should_log = embodied.when.Clock(args.log_every)
    # if need_debug_env_step(args):
    #     should_log_timer = embodied.when.Clock(args.log_env_step.every)
    if not args.mixed_precision:
        set_global_policy("float32")
    win_rates = []

    logdir = Path(args.logdir)
    checkpoint = Checkpoint(logdir / "checkpoint.ckpt")
    checkpoint.step = step
    checkpoint.real_step = real_env_step
    checkpoint.agent = agent

    if args.from_checkpoint:
        checkpoint.load(args.from_checkpoint, strict=True)

    def callback(obs, env_addrs):
        metrics.scalar("parallel/ep_starts", obs["is_first"].sum(), agg="sum")
        metrics.scalar("parallel/ep_ends", obs["is_last"].sum(), agg="sum")

        obs_steps = np.zeros((args.actor_batch), dtype=np.int32)
        for i, a in enumerate(env_addrs):
            if obs["is_first"][i]:
                abandoned = not dones.get(a, True)
                metrics.scalar("parallel/episode_abandoned", int(abandoned), agg="sum")
            dones[a] = obs["is_last"][i]
            obs_steps[i] = -1

        states = [allstates[a] for a in env_addrs]
        states = embodied.treemap(lambda *xs: list(xs), *states)

        # take action from ALL? policy
        act, states, info = agent.policy(obs, states, step=obs_steps, mode="test")
        # logger.add(test_metrics.result(), "test")
        act["reset"] = obs["is_last"].copy()

        for i, a in enumerate(env_addrs):
            allstates[a] = embodied.treemap(lambda x: x[i], states)

        # eps_cnt.increment(obs["is_last"].sum())
        step.increment(args.actor_batch)
        real_env_step.increment(
            (~obs["is_read_step"]).sum() if "is_read_step" in obs else args.actor_batch
        )
        metrics.scalar("parallel/ep_states", len(allstates))

        trans = {**obs, **act}
        now = time.time()
        for i, a in enumerate(env_addrs):
            tran = {k: v[i].copy() for k, v in trans.items()}

            if tran["is_first"]:
                scalars.pop(a, None)
                # videos.pop(a, None)
                # vidstreams.pop(a, None)

            [scalars[a][k].append(v) for k, v in tran.items() if v.size == 1]
            # if a in vidstreams or len(vidstreams) < args.log_video_streams:
            #     vidstreams[a] = now
            #     [videos[a][k].append(tran[k]) for k in args.log_keys_video if k != ""]

        # for a, last_add in list(vidstreams.items()):
        #     if now - last_add > args.log_video_timeout:
        #         print(f"Dropping video stream due to timeout ({now - last_add:.1f}s).")
        #         del vidstreams[a]
        #         del videos[a]

        for i, a in enumerate(env_addrs):
            if not trans["is_last"][i]:
                continue

            ep = scalars.pop(a)

            # if a in vidstreams:
            #     ep.update(videos.pop(a))
            #     del vidstreams[a]

            ep = {k: convert(v) for k, v in ep.items()}
            logger.add(
                {
                    "score": sum(ep["reward"]),
                },
                prefix="test_episodes/",
            )
            if done_cal_win_rate(win_rates, sum(ep["reward"]), eps_cnt, config):
                # checkpoint.save()
                cprint(f"eval.done | episodes={args.eval_eps}", "green")
                raise Exception("DONE")
            else:
                win_rate = WIN_REWARDS[config.task.split("_")[1]]
                if len(win_rates) % 100 == 0:
                    event_logger.info(
                        "eval.progress | episodes={} | matching_rate={:.4f}",
                        len(win_rates),
                        np.mean(np.array(win_rates) == win_rate),
                    )
                # exit()

            stats = {}
            for key in args.log_keys_video:
                if key == "":
                    continue
                if key in ep:
                    stats[f"policy_{key}"] = ep[key]

            for key, value in ep.items():
                if not args.log_zeros and key not in nonzeros and np.all(value == 0):
                    continue
                nonzeros.add(key)
                if re.match(args.log_keys_sum, key):
                    stats[f"sum_{key}"] = ep[key].sum()
                if re.match(args.log_keys_mean, key):
                    stats[f"mean_{key}"] = ep[key].mean()
                if re.match(args.log_keys_max, key):
                    stats[f"max_{key}"] = ep[key].max(0).mean()
            metrics.add(stats, prefix="stats")

        if should_log():
            logger.add(metrics.result())
        return act

    _, initial, info = agent.policy(
        dummy_data(agent.agent.obs_space, (args.actor_batch,)),
        mode="test",  # type: ignore
    )
    initial = embodied.treemap(lambda x: x[0], initial)
    allstates = defaultdict(lambda: initial)
    nonzeros = set()
    # vidstreams = {}
    dones = {}

    server = distr.Server(
        callback, args.actor_port, args.ipv6, args.actor_batch, args.actor_threads
    )
    timer.wrap("server", server, ["_step", "_work"])
    server.run()


def parallel_env_eval(replica_id, make_env, args, timer=None):
    # TODO: Optionally write NPZ episodes.
    configure_logging()
    assert replica_id >= 0, replica_id
    rid = replica_id
    event_logger.debug("env.create | env_id={} | mode=eval", rid)
    env = make_env()
    timer and timer.wrap("env", env, ["step"])  # type: ignore
    addr = f"{args.actor_host}:{args.actor_port}"
    actor = distr.Client(
        addr,
        replica_id,
        args.ipv6,
        timeout=distr.resolve_actor_timeout(args),
        max_reconnect=MAX_RECONNECT_LOAD,
    )
    done = True
    act = None
    start = time.time()
    score = length = count = 0
    # is_debug_env_step() = args.log_timer > 0

    while True:
        if done:
            act = {k: v.sample() for k, v in env.act_space.items()}
            act["reset"] = True
            score, length = 0, 0

        start_timer = time.time()
        obs = env.step(act)
        end_timer = time.time()

        obs = {k: np.asarray(v) for k, v in obs.items()}
        score += obs["reward"]
        length += 1
        done = obs["is_last"]
        if done:
            event_logger.debug(
                "env.episode | env_id={} | mode=eval | length={} | score={:.4f}",
                rid,
                length,
                score,
            )
            # eps_cnt.increment()

        promise = actor(obs)
        try:
            act = promise()
            act = {k: v for k, v in act.items() if not k.startswith("log_")}
        except distr.ReconnectError:
            event_logger.warning(
                "env.reconnect | env_id={} | action=restart_episode", rid
            )
            done = True

        except distr.RemoteError as e:
            event_logger.error(
                "env.shutdown | env_id={} | reason=agent_error | error={}", rid, e
            )
            sys.exit(0)

        count += 1
        now = time.time()
        if now - start >= 60:
            fps = count / (now - start)
            event_logger.debug(
                "env.throughput | env_id={} | fps={:.1f}", rid, fps
            )
            start = now
            count = 0
