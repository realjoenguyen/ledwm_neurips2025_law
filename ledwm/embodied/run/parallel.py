import pathlib
import matplotlib.pyplot as plt
import jax
import re
import sys
import threading
import time
from collections import OrderedDict, defaultdict
from multiprocessing import Lock, Value
from typing import TYPE_CHECKING, Dict, List, Optional

from tqdm import tqdm

from ledwm.common import get_multihot_image_from_pos, log_image
from ledwm.embodied.envs.LWMSent import DEAD_ID

# from jaxutils import draw_cont_prob_hist, draw_precision_recall_curve
from termcolor import cprint
from torch.utils.data import DataLoader, IterableDataset

# from sklearn.metrics import f1_score, precision_recall_curve
import wandb
from ledwm.embodied.replay.generic import (
    GenericReplay,
    ReplayProgress,
    consume_replay_sample_stats,
    convert2uuid,
)
from ledwm.embodied.core.counter import Counter
from ledwm.embodied.core.path import Path
from ledwm.embodied.core.checkpoint import Checkpoint
from ledwm.embodied.core.timer import Timer
from ledwm.embodied.core.metrics import Metrics
from ledwm.embodied.core.when import Clock, Once
from ledwm.embodied.core.basics import treemap, convert
from ledwm.embodied.core import distr
from ledwm.embodied.run.action_rpc import compact_action_for_rpc
from ledwm.embodied.run.async_replay import AsyncReplayAdder
from ledwm.embodied.run.async_worker import AsyncWorker
from ledwm.embodied.run.env_worker import get_env_address, parallel_env
from ledwm.embodied.run.timing import (
    long_operation_progress,
    make_timing,
    maybe_report_to_logger,
    record_timing_breakdown,
)
from ledwm.logging_setup import logger as event_logger

# from ledwm.embodied.core.batcher import Batcher
if TYPE_CHECKING:
    from ledwm.embodied.core.logger import Logger
    from ledwm.embodied.replay.replays import Uniform
    from ledwm.jaxagent import JAXAgent
    from ledwm.embodied.core.config import Config

import numpy as np

from ledwm.embodied.core.WandBOutput import make_table_data_text
from ledwm.embodied.replay.Prioritized import PrioritizedReplay, PrioritizedSampler
from ledwm.embodied.run.smoothing import ReplayEps
from ledwm.train_signature import canonical_reward_weight_keys

start_parallel = 0
# share variable among multiple processes
total_step_time = Value("d", 0.0)
total_steps = Value("i", 0)
total_eps = Value("i", 0)
lock = Lock()
PRIORITY_KEY = "priority_loss_per_batch"


def _wait_for_first_agent_update(agent, sleep=time.sleep):
    while agent.updates.value == 0:
        sleep(0.25)


def _replay_startup_detail(replay):
    limiter = getattr(replay, "limiter", None)
    minimum = getattr(limiter, "minimum", "unknown")
    return (
        f"replay sequences={len(replay)}/{minimum}; "
        "waiting for data, learner JIT has not started"
    )


def _warn_replay_starvation(avail_ratio, already_starved, opt_step, log=event_logger):
    starved = avail_ratio <= 0
    if starved and not already_starved:
        log.warning(
            "replay.starved | avail={:.2%} | opt_step={} | "
            "state=waiting_for_insert_credit",
            avail_ratio,
            opt_step,
        )
    return starved


def _save_best_eval_if_improved(checkpoint, agent, eval_win_rate, margin=0):
    """Persist a best-eval checkpoint without truncating fractional win rates."""
    best = checkpoint.best_eval_win_rate
    if float(eval_win_rate) <= float(best.value) + float(margin):
        return False
    best.load(float(eval_win_rate))
    checkpoint.best_eval_agent = agent
    checkpoint.save()
    return True


def _train_reward_weights(train_replay, config, reward_keys=None):
    if config.replay.imbalance != "balanced_weight":
        return None
    keys = reward_keys
    if keys is None:
        keys = canonical_reward_weight_keys(config)
    if keys is None:
        keys = train_replay.imbalanced_rewards
    return OrderedDict(
        (key, train_replay.reward_balanced_weight(key)) for key in keys
    )


def id2sent_from_env_cache(env_cache):
    # sent2id: from sent to id
    # return id2sent: from id to sent
    sent2id = env_cache["sent_ids"]
    id2sent = {v: k for k, v in sent2id.items()}
    return id2sent


def draw_hist_whole_dataset(sampled_keys, dataset: dict, logdir):
    keys_copy = list(dataset.keys())
    key2order = {k: i for i, k in enumerate(keys_copy)}
    orders = [key2order[k] for k in sampled_keys]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(
        orders,
        bins=20,
        range=(0, len(dataset)),
        alpha=0.7,
        color="blue",
        edgecolor="black",
    )
    ax.set_xlabel(f"Key Value (0 to {len(dataset)})", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.grid(axis="y", linestyle="-", alpha=0.7)
    return fig


def draw_hist_priority(priorities):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(priorities, bins=100, alpha=0.7, color="blue", edgecolor="black")
    ax.set_xlabel("Priority", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.grid(axis="y", linestyle="-", alpha=0.7)
    return fig


def draw_dist_whole_dataset(num_samples, logdir):
    assert num_samples is not None
    dataset = range(1, len(num_samples) + 1)  # Generate dataset indices
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    # ax.plot(dataset, num_samples, linestyle="-", linewidth=2)
    # do points instead
    ax.plot(
        dataset,
        num_samples,
        linestyle="",
        marker="o",
        markersize=5,
        color="blue",
        alpha=0.7,
    )
    ax.set_xlabel("Dataset", fontsize=14)
    ax.set_ylabel("# Sampled", fontsize=14)
    ax.set_title("Sampling Distribution", fontsize=16)
    ax.legend(fontsize=12)
    ax.grid(alpha=0.3)
    path = pathlib.Path(logdir) / "sample_dist.png"
    fig.savefig(str(path))
    cprint(f"report.sample_distribution_saved | path={path}", "green")

    return fig


def parallel(
    agent: "JAXAgent",
    train_replay: "Uniform | PrioritizedReplay | ReplayEps",
    logger: "Logger",
    make_train_env,
    args,
    env_cache=None,
    make_eval_env=None,
    make_test_env=None,
    eval_replay: "Optional[Uniform|ReplayEps]" = None,
    test_replay: "Optional[ReplayEps]" = None,
    config=None,
):
    logdir = Path(args.logdir)
    checkpoint = Checkpoint(logdir / "checkpoint_1.ckpt")
    env_step = logger.step
    real_env_step = Counter()
    opt_step = Counter()
    checkpoint.step = env_step
    checkpoint.real_step = real_env_step
    checkpoint.agent = agent
    checkpoint.opt_step = opt_step
    checkpoint.replay_progress = ReplayProgress(train_replay)
    checkpoint.best_eval_win_rate = Counter()
    checkpoint.best_eval_agent = agent

    # if args.from_checkpoint != "":
    #     cp_path = args.from_checkpoint
    # elif args.from_checkpoint != "":
    #     cp_path = args.from_checkpoint
    # else:
    #     cp_path = None

    if args.from_checkpoint != "":
        cp_path = pathlib.Path(args.from_checkpoint)
        assert cp_path.exists(), cp_path
        if "checkpoint" not in cp_path.name:
            cp_path = cp_path.parent / f"checkpoint_1.ckpt"
        checkpoint.load(cp_path, skip_key=args.skip_key, strict=True)

    if args.load_checkpoint:
        checkpoint.load_or_save()

    if wandb.run is not None:
        cprint(f"run.wandb_resume | step={wandb.run.step}")
        wandb_step = wandb.run.step
        env_step.load(wandb_step)
        checkpoint.step.load(wandb_step)
        logger.update_step(wandb_step)
        event_logger.info(
            "run.step_sync | env_step={} | checkpoint_step={}",
            env_step,
            checkpoint.step,
        )
        event_logger.info("run.logger | value={}", logger)

    cprint(
        f"run.resume | env_step={env_step} | real_env_step={real_env_step}",
        "red",
    )
    assert logger.step >= wandb.run.step if wandb.run is not None else True

    timer = Timer()
    timer.wrap("agent", agent, ["policy", "train", "report", "save"])
    timer.wrap("replay", train_replay, ["add", "save"])
    timer.wrap("logger", logger, ["write"])
    # usage = Usage(args.trace_malloc)

    workers = []
    event_logger.info("run.workers_start")
    global start_parallel
    start_parallel = time.time()
    assert config is not None, "config is None"
    train_env_ids = list(range(config.envs.amount))
    worker_addr2type = {get_env_address(i): "train" for i in train_env_ids}
    # assert env_cache is not None, "env_cache is None"
    if env_cache is not None:
        id2sent = id2sent_from_env_cache(env_cache)
    else:
        id2sent = None

    # Start policy initialization before launching the env-process stampede.
    # Spawned clients can queue their first request while the policy compiles,
    # and the server can serve full batches before the slowest env is ready.
    actor_worker = distr.Thread(
        parallel_actor,
        env_step,
        real_env_step,
        opt_step,
        agent,
        train_replay,
        logger,
        args,
        worker_addr2type,
        checkpoint,
        eval_replay,
        test_replay,
    )
    actor_worker.start()
    workers.append(actor_worker)
    event_logger.info("actor.worker_start | phase=before_envs")

    if make_eval_env is not None:
        assert args.script in [
            "parallel_train_eval",
            "parallel_train_eval_test",
            "finetune_wm",
        ]
        eval_env_ids = list(
            range(len(train_env_ids), len(train_env_ids) + config.num_eval_envs)
        )
        worker_addr2type.update({get_env_address(i): "eval" for i in eval_env_ids})

        workers += _spawn_env_workers(
            eval_env_ids,
            make_eval_env,
            args,
            worker_addr2type,
            timer,
            "parallel_env_eval",
            thread_env_id=1,
        )

        if "test" in args.script:
            event_logger.debug("env.group_init | mode=test")
            assert make_test_env is not None, "make_test_env is None"
            test_env_ids = list(
                range(
                    len(worker_addr2type), len(worker_addr2type) + config.num_eval_envs
                )
            )
            worker_addr2type.update({get_env_address(i): "test" for i in test_env_ids})

            workers += _spawn_env_workers(
                test_env_ids,
                make_test_env,
                args,
                worker_addr2type,
                timer,
                "parallel_env_test",
                thread_env_id=1,
            )

    # TRAIN ENVS
    workers += _spawn_env_workers(
        train_env_ids,
        make_train_env,
        args,
        worker_addr2type,
        timer,
        "parallel_env",
        thread_env_id=0,
    )

    if "finetune_wm" not in args.script:
        workers.append(
            distr.Thread(
                parallel_learner,
                env_step,
                real_env_step,
                opt_step,
                agent,
                train_replay,
                logger,
                # timer,
                # usage,
                checkpoint,
                args,
                id2sent,
                config,
            )
        )

    if make_eval_env is not None:
        workers.append(
            distr.Thread(
                parallel_eval,
                env_step,
                real_env_step,
                opt_step,
                agent,
                eval_replay,
                logger,
                timer,
                args,
                config,
                "eval",
                id2sent,
            )
        )

    if make_test_env is not None:
        workers.append(
            distr.Thread(
                parallel_eval,
                env_step,
                real_env_step,
                opt_step,
                agent,
                test_replay,
                logger,
                timer,
                args,
                config,
                "test",
                id2sent,
            )
        )

    distr.run(workers)


def _spawn_env_workers(
    env_ids, make_env, args, worker_addr2type, timer, name_prefix, thread_env_id
):
    """Build the env workers for `env_ids`.

    A single env runs in-process as a Thread (using `thread_env_id`, preserving
    the historical local-id behavior); multiple envs each get their own spawned
    Process, started immediately. Returns the list of workers to append.
    """
    if len(env_ids) == 1:
        return [
            distr.Thread(
                parallel_env, thread_env_id, make_env, args, worker_addr2type, timer
            )
        ]
    workers = []
    for i in env_ids:
        worker = distr.Process(
            parallel_env,
            i,
            make_env,
            args,
            worker_addr2type,
            name=f"{name_prefix}_{i}",
        )
        worker.start()
        workers.append(worker)
    return workers


def parallel_actor(
    env_step: "Counter",  # logger.step
    real_env_step: "Counter",
    opt_step: "Counter",
    agent: "JAXAgent",
    train_replay: "ReplayEps",
    logger: "Logger",
    # timer: "Timer",
    args,
    worker_addr2type: Dict[int, bytes],
    checkpoint: "Checkpoint",
    eval_replay: "Uniform|None" = None,
    test_replay: "Uniform|None" = None,
):
    cprint("actor.start", "green")
    metrics = Metrics()
    # to store all steps in one eps - for each env
    scalars_eps = defaultdict(lambda: defaultdict(list))
    error_cnt = 0

    should_log_video = False
    videos = None
    if args.log_video:
        should_log_video = Clock(args.log_video_every)
        videos = defaultdict(lambda: defaultdict(list))  # addr2video

    cprint(
        f"actor.video_logging | enabled={str(bool(should_log_video)).lower()}",
        "green" if should_log_video else "red",
    )

    should_log = Clock(args.log_every)
    train_wins_sofar = []
    test_scores_sofar = None if test_replay is None else []
    eval_wins_sofar = None if eval_replay is None else []
    WIN_RATES = [0, 0.25, 0.5, 0.75, 1]
    # checkpoint_win_rates = [0]
    # save checkpoint

    save_checkpoints = "finetune" not in args.script
    if save_checkpoints:
        for win_rate in WIN_RATES:
            path = pathlib.Path(args.logdir) / f"checkpoint_{win_rate}.ckpt"
            if path.exists():
                cprint(f"checkpoint.exists | path={path} | found=true", "red")

        # save checkpoint_0
        val = WIN_RATES.pop(0)
        assert val == 0, val
        path = pathlib.Path(args.logdir) / f"checkpoint_{val}.ckpt"
        if not path.exists():
            checkpoint.save(path)
        else:
            cprint(f"checkpoint.save_skipped | path={path} | reason=exists", "red")

    finetune_wm = "finetune_wm" in args.script
    num_eps_so_far = 0
    if finetune_wm:
        cp_dir = pathlib.Path(args.from_checkpoint).parent
        # assert paths exist
        for win_rate in WIN_RATES:
            path = cp_dir / f"checkpoint_{win_rate}.ckpt"
            if win_rate == 1 and not path.exists():
                path = cp_dir / "checkpoint.ckpt"
            assert path.exists(), (cp_dir, path)

        val = WIN_RATES.pop(0)
        path = cp_dir / f"checkpoint_{val}.ckpt"
        checkpoint.load(path)

    actor_timing = make_timing("actor", args)
    action_space = getattr(agent.agent, "act_space", None)
    keep_policy_state_on_device = (
        getattr(args, "keep_policy_state_on_device", True)
        and len(getattr(agent, "policy_devices", [])) == 1
    )
    async_replay_enabled = getattr(args, "async_replay_add", False) and not finetune_wm
    async_replay_adder = None
    if async_replay_enabled:
        async_replay_adder = AsyncReplayAdder(
            maxsize=getattr(args, "async_replay_add_queue", 4096)
        )
        cprint(
            f"actor.async_replay_add | enabled=true | "
            f"queue_size={getattr(args, 'async_replay_add_queue', 4096)}",
            "yellow",
        )
    elif getattr(args, "async_replay_add", False):
        cprint(
            "actor.async_replay_add | enabled=false | reason=finetune_wm",
            "yellow",
        )

    async_actor_postprocess_enabled = (
        getattr(args, "async_actor_postprocess", False) and not finetune_wm
    )
    actor_postprocess_worker = None
    if async_actor_postprocess_enabled:
        actor_postprocess_worker = AsyncWorker(
            "actor_postprocess",
            maxsize=getattr(args, "async_actor_postprocess_queue", 2),
        )
        cprint(
            "actor.async_postprocess | enabled=true | "
            f"queue_size={getattr(args, 'async_actor_postprocess_queue', 2)}",
            "yellow",
        )
    elif getattr(args, "async_actor_postprocess", False):
        cprint(
            "actor.async_postprocess | enabled=false | reason=finetune_wm",
            "yellow",
        )

    def record_actor_timing(timing_values, policy_timing, env_count):
        if actor_timing is None:
            return
        actor_timing.event()
        for key, value in timing_values.items():
            actor_timing.record(key, value)
        record_timing_breakdown(actor_timing, "policy", policy_timing)
        actor_timing.increment("env_steps", env_count)
        maybe_report_to_logger(actor_timing, logger, "actor_timing")

    def post_policy_callback(
        obs,
        act,
        env_addrs,
        policy_timing,
        callback_start,
        action_timings,
        episode_abandoned,
        ep_states,
        real_env_step_value,
        ready=None,
        async_postprocess=False,
    ):
        nonlocal error_cnt
        if ready is not None:
            ready.wait()

        post_start = time.time()
        train_replay_add_time = 0.0
        eval_replay_add_time = 0.0
        test_replay_add_time = 0.0
        post_trans_time = 0.0
        post_episode_accum_time = 0.0
        post_video_time = 0.0
        post_video_gc_time = 0.0
        post_episode_log_time = 0.0
        post_logger_time = 0.0

        if episode_abandoned:
            metrics.scalar("parallel/episode_abandoned", episode_abandoned, agg="sum")
        metrics.scalar("parallel/ep_states", ep_states)

        trans_start = time.time()
        trans = {**obs, **act}  # obs, act is the next action given this obs
        post_trans_time += time.time() - trans_start
        time_now = time.time()

        for i, addr in enumerate(env_addrs):
            trans_start = time.time()
            tran = {k: v[i].copy() for k, v in trans.items()}
            post_trans_time += time.time() - trans_start

            accum_start = time.time()
            if tran["is_first"]:
                scalars_eps.pop(addr, None)
                if args.log_video:
                    assert videos is not None
                    videos.pop(addr, None)
                    vidstreams.pop(addr, None)

            [scalars_eps[addr][k].append(v) for k, v in tran.items() if v.size == 1]
            post_episode_accum_time += time.time() - accum_start

            if args.log_video:
                video_start = time.time()
                assert isinstance(should_log_video, Clock)
                if addr in vidstreams or (
                    len(vidstreams) < args.log_video_streams
                    and tran["time_step"] == 0
                    and should_log_video()
                ):
                    vidstreams[addr] = time_now
                    raw_image = get_multihot_image_from_pos(
                        tran["entity_pos"], tran["avatar_pos"]
                    )
                    image = log_image(
                        raw_image,
                        tran["time_step"],
                        tran["action"],
                        tran["reward"],
                        tran["is_last"],
                        tran["log_reward_pred"],
                        tran["log_cont_pred"],
                    )
                    assert videos is not None
                    videos[addr]["image"].append(image)
                    if tran["is_last"]:
                        for _ in range(1, 3):
                            videos[addr]["image"].append(image)
                post_video_time += time.time() - video_start

            trans_start = time.time()
            tran = {k: v for k, v in tran.items() if not k.startswith("log_")}
            post_trans_time += time.time() - trans_start

            if worker_addr2type[addr] == "train":
                add_start = time.time()
                if async_replay_adder is None:
                    train_replay.add(tran.copy(), worker=addr)  # Blocks when rate limited.
                else:
                    async_replay_adder.add(train_replay, tran.copy(), worker=addr)
                train_replay_add_time += time.time() - add_start

            elif worker_addr2type[addr] == "eval":
                assert eval_replay is not None
                add_start = time.time()
                if async_replay_adder is None:
                    eval_replay.add(tran.copy(), worker=addr, training=False)
                else:
                    async_replay_adder.add(
                        eval_replay, tran.copy(), worker=addr, training=False
                    )
                eval_replay_add_time += time.time() - add_start

            elif worker_addr2type[addr] == "test":
                assert test_replay is not None
                add_start = time.time()
                if async_replay_adder is None:
                    test_replay.add(tran.copy(), worker=addr, training=False)
                else:
                    async_replay_adder.add(
                        test_replay, tran.copy(), worker=addr, training=False
                    )
                test_replay_add_time += time.time() - add_start
            else:
                raise ValueError(f"Unknown env type: {worker_addr2type[addr]}")

        if args.log_video:
            video_gc_start = time.time()
            assert videos is not None
            for addr, last_add_time in list(vidstreams.items()):
                if time_now - last_add_time > args.log_video_timeout:
                    event_logger.warning(
                        "actor.video_stream_dropped | env_id={} | "
                        "reason=timeout | idle={:.1f}s",
                        addr,
                        time_now - last_add_time,
                    )
                    del vidstreams[addr]
                    del videos[addr]
            post_video_gc_time += time.time() - video_gc_start

        episode_log_start = time.time()
        for i, addr in enumerate(env_addrs):
            if not trans["is_last"][i]:
                continue

            if addr not in scalars_eps:
                error_cnt += 1
                cprint(
                    f"actor.episode_metrics_missing | env_id={addr} | "
                    f"count={error_cnt}",
                    "red",
                )

            ep_addr = scalars_eps.pop(addr)
            if args.log_video:
                assert videos is not None
                if addr in vidstreams:
                    ep_addr.update(videos.pop(addr))
                    del vidstreams[addr]

            ep_addr = {k: convert(v) for k, v in ep_addr.items()}
            logger.add(
                {
                    "length": len(ep_addr["reward"]) - 1,
                    "score": sum(ep_addr["reward"]),
                },
                prefix=f"episode_{worker_addr2type[addr]}",
            )

            if worker_addr2type[addr] == "train":
                score = sum(ep_addr["reward"])
                if not np.isnan(score):
                    train_wins_sofar.append(score > 0)

                if len(train_wins_sofar) >= args.eval_eps:
                    train_win_rate = np.mean(train_wins_sofar)
                    with tqdm.external_write_mode():
                        cprint(
                            f"actor.win_rate | mode=train | "
                            f"value={train_win_rate:.4f} | "
                            f"episodes={len(train_wins_sofar)}",
                        )
                    logger.add({"win_rate": train_win_rate}, prefix="train")
                    train_wins_sofar.clear()

                    if (
                        save_checkpoints
                        and len(WIN_RATES) > 0
                        and train_win_rate > WIN_RATES[0]
                    ):
                        path = (
                            pathlib.Path(args.logdir) / f"checkpoint_{WIN_RATES[0]}.ckpt"
                        )
                        WIN_RATES.pop(0)
                        if not path.exists():
                            checkpoint.save(path)
                        else:
                            cprint(
                                f"checkpoint.save_skipped | path={path} | "
                                "reason=exists",
                                "red",
                            )

            elif worker_addr2type[addr] == "eval":
                score = sum(ep_addr["reward"])
                assert eval_wins_sofar is not None
                if not np.isnan(score):
                    eval_wins_sofar.append(score > 0)

                if len(eval_wins_sofar) >= args.eval_eps:
                    win_rate = np.mean(eval_wins_sofar)
                    with tqdm.external_write_mode():
                        cprint(
                            f"actor.win_rate | mode=eval | value={win_rate:.4f} | "
                            f"episodes={len(eval_wins_sofar)}",
                            "green",
                        )
                    eval_win_rate = np.mean(eval_wins_sofar)
                    logger.add({"win_rate": eval_win_rate}, prefix="eval")

                    eval_wins_sofar.clear()
                    MARGIN = 0
                    assert checkpoint is not None
                    assert checkpoint.best_eval_win_rate is not None
                    if _save_best_eval_if_improved(
                        checkpoint, agent, eval_win_rate, MARGIN
                    ):
                        cprint(
                            f"actor.best_eval | win_rate={eval_win_rate:.4f}",
                            "green",
                        )

            elif worker_addr2type[addr] == "test":
                score = sum(ep_addr["reward"])
                assert test_scores_sofar is not None
                if not np.isnan(score):
                    test_scores_sofar.append(score > 0)
                if len(test_scores_sofar) >= args.eval_eps:
                    test_win_rate = np.mean(test_scores_sofar)
                    with tqdm.external_write_mode():
                        cprint(
                            f"actor.win_rate | mode=test | "
                            f"value={test_win_rate:.4f} | "
                            f"episodes={len(test_scores_sofar)}",
                            "green",
                        )
                    logger.add({"win_rate": np.mean(test_scores_sofar)}, prefix="test")
                    test_scores_sofar.clear()

            if worker_addr2type[addr] == "train":
                logger.add({"real_step": real_env_step_value})
                stats = {}
                if args.log_video:
                    for key in args.log_keys_video:
                        if key in ep_addr:
                            tag = (
                                f"train/win/policy_{key}"
                                if ep_addr["reward"][-1] == 1
                                else f"train/lose/policy_{key}"
                            )
                            stats[tag] = ep_addr[key]

                for key, value in ep_addr.items():
                    if (
                        not args.log_zeros
                        and key not in nonzeros
                        and np.all(value == 0)
                    ):
                        continue
                    nonzeros.add(key)
                    if re.match(args.log_keys_sum, key):
                        stats[f"sum_{key}"] = ep_addr[key].sum()
                    if re.match(args.log_keys_mean, key):
                        stats[f"mean_{key}"] = ep_addr[key].mean()
                    if re.match(args.log_keys_max, key):
                        stats[f"max_{key}"] = ep_addr[key].max(0).mean()
                metrics.add(stats, prefix="stats")

            elif worker_addr2type[addr] == "eval":
                stats = {}
                if args.log_video:
                    for key in args.log_keys_video:
                        if key in ep_addr:
                            tag = (
                                f"eval/win/policy_{key}"
                                if ep_addr["reward"][-1] == 1
                                else f"eval/lose/policy_{key}"
                            )
                            stats[tag] = ep_addr[key]
                metrics.add(stats, prefix="stats")

            elif worker_addr2type[addr] == "test":
                stats = {}
                if args.log_video:
                    for key in args.log_keys_video:
                        if key in ep_addr:
                            tag = (
                                f"test/win/policy_{key}"
                                if ep_addr["reward"][-1] == 1
                                else f"test/lose/policy_{key}"
                            )
                            stats[tag] = ep_addr[key]
                metrics.add(stats, prefix="stats")
        post_episode_log_time += time.time() - episode_log_start

        logger_start = time.time()
        if should_log():
            logger.add(metrics.result())
        post_logger_time += time.time() - logger_start

        post_total = time.time() - post_start
        callback_time = (
            action_timings["callback"]
            if async_postprocess
            else time.time() - callback_start
        )
        action_known_time = sum(
            action_timings.get(key, 0.0)
            for key in (
                "pre_policy",
                "state_gather",
                "finetune_sync",
                "policy",
                "state_scatter",
                "counter",
                "action_rpc",
                "post_enqueue",
            )
        )
        if not async_postprocess:
            action_known_time += post_total
        post_known_time = (
            post_trans_time
            + post_episode_accum_time
            + post_video_time
            + post_video_gc_time
            + train_replay_add_time
            + eval_replay_add_time
            + test_replay_add_time
            + post_episode_log_time
            + post_logger_time
        )
        timing_values = {
            **action_timings,
            "callback": callback_time,
            "postprocess": post_total,
            "post_trans": post_trans_time,
            "post_episode_accum": post_episode_accum_time,
            "post_video": post_video_time,
            "post_video_gc": post_video_gc_time,
            "train_replay_add": train_replay_add_time,
            "eval_replay_add": eval_replay_add_time,
            "test_replay_add": test_replay_add_time,
            "post_episode_log": post_episode_log_time,
            "post_logger": post_logger_time,
            "other": max(0.0, callback_time - action_known_time),
            "post_other": max(0.0, post_total - post_known_time),
        }
        record_actor_timing(timing_values, policy_timing, len(env_addrs))

    def callback(obs, env_addrs):
        nonlocal num_eps_so_far
        callback_start = time.time()
        if actor_postprocess_worker is not None:
            actor_postprocess_worker.raise_if_failed()
        if async_replay_adder is not None:
            async_replay_adder.raise_if_failed()

        pre_policy_start = time.time()
        obs_steps = (
            np.zeros((args.actor_batch), dtype=np.int32)
            if args.script == "parallel_train_eval_test"
            else None
        )
        episode_abandoned = 0
        for i, addr in enumerate(env_addrs):
            if obs["is_first"][i]:
                episode_abandoned += int(not dones.get(addr, True))
            dones[addr] = obs["is_last"][i]
            if obs_steps is not None:
                obs_steps[i] = (
                    opt_step.value if worker_addr2type[addr] == "train" else -1
                )
        pre_policy_time = time.time() - pre_policy_start

        state_gather_start = time.time()
        states = [allstates[a] for a in env_addrs]
        states = treemap(lambda *xs: list(xs), *states)
        state_gather_time = time.time() - state_gather_start

        finetune_sync_start = time.time()
        if finetune_wm:
            if len(train_replay) - num_eps_so_far > args.num_eps_each_policy:
                if len(WIN_RATES) == 0:
                    cprint("actor.policy_collection_done", "green")
                    train_replay.save()
                    checkpoint.save()
                    sys.exit(0)
                    return

                next_win_rate = WIN_RATES.pop(0)
                cprint(
                    f"actor.policy_switch | episodes_loaded="
                    f"{len(train_replay) - num_eps_so_far} | "
                    f"checkpoint_win_rate={next_win_rate}",
                    "green",
                )
                num_eps_so_far = len(train_replay)

                path = cp_dir / f"checkpoint_{next_win_rate}.ckpt"  # type: ignore
                if next_win_rate == "final" and not path.exists():
                    path = cp_dir / "checkpoint.ckpt"  # type: ignore
                checkpoint.load(path)
        finetune_sync_time = time.time() - finetune_sync_start

        policy_start = time.time()
        act, states, info = agent.policy(
            obs,
            states,
            step=obs_steps,
            mode="train",
            return_state_to_host=not keep_policy_state_on_device,
        )  # (env, ...)
        policy_time = time.time() - policy_start
        policy_timing = getattr(agent, "last_policy_timing", None)

        assert not np.any(np.isnan(act["action"])), act["action"]

        act["reset"] = obs["is_last"].copy()
        state_scatter_start = time.time()
        for i, addr in enumerate(env_addrs):
            allstates[addr] = treemap(lambda x: x[i], states)
        state_scatter_time = time.time() - state_scatter_start

        counter_start = time.time()
        env_step.increment(args.actor_batch)  # += actor_batch == env
        real_env_step.increment(
            (~obs["is_read_step"]).sum() if "is_read_step" in obs else args.actor_batch
        )
        real_env_step_value = real_env_step.value
        counter_time = time.time() - counter_start

        action_rpc_start = time.time()
        ACTION_KEYS = ["action", "reset"]
        rpc_act = {k: v for k, v in act.items() if k in ACTION_KEYS}
        if getattr(args, "compact_action_rpc", True):
            rpc_act = compact_action_for_rpc(rpc_act, action_space)
        action_rpc_time = time.time() - action_rpc_start

        action_timings = {
            "pre_policy": pre_policy_time,
            "state_gather": state_gather_time,
            "finetune_sync": finetune_sync_time,
            "policy": policy_time,
            "state_scatter": state_scatter_time,
            "counter": counter_time,
            "action_rpc": action_rpc_time,
        }
        if actor_postprocess_worker is None:
            action_timings["post_enqueue"] = 0.0
            action_timings["action_path"] = time.time() - callback_start
            post_policy_callback(
                obs,
                act,
                env_addrs,
                policy_timing,
                callback_start,
                action_timings,
                episode_abandoned,
                len(allstates),
                real_env_step_value,
            )
        else:
            ready = threading.Event()
            submit_start = time.time()
            try:
                actor_postprocess_worker.submit(
                    post_policy_callback,
                    obs,
                    act,
                    env_addrs,
                    policy_timing,
                    callback_start,
                    action_timings,
                    episode_abandoned,
                    len(allstates),
                    real_env_step_value,
                    ready,
                    True,
                )
                action_timings["post_enqueue"] = time.time() - submit_start
                action_timings["callback"] = time.time() - callback_start
                action_timings["action_path"] = action_timings["callback"]
            finally:
                ready.set()
        return rpc_act

    cprint(f"actor.policy_init | batch_size={args.actor_batch}", "green")
    # The first call traces and compiles the policy. It can be quiet for minutes.
    with long_operation_progress(
        "actor first policy JIT",
        f"batch={args.actor_batch}; tracing/compiling on policy device",
    ):
        # outs, state. Get initial as policy state
        _, initial, info = agent.policy(
            dummy_data(agent.agent.obs_space, (args.actor_batch,)),  # type: ignore
            step=opt_step.value if args.opt_step else None,
            mode="train",
            return_state_to_host=not keep_policy_state_on_device,
        )

    initial = treemap(lambda x: x[0] if x.size > 0 else x, initial)
    allstates = defaultdict(lambda: initial)
    nonzeros = set()
    vidstreams = {}
    dones = {}

    server = distr.Server(
        callback, args.actor_port, args.ipv6, args.actor_batch, args.actor_threads
    )
    # timer.wrap("server", server, ["_step", "_work"])
    try:
        server.run()
    finally:
        try:
            if actor_postprocess_worker is not None:
                actor_postprocess_worker.close()
        finally:
            if async_replay_adder is not None:
                async_replay_adder.close()


# def get_step_for_envs()


class Dataset(IterableDataset):
    def __init__(self, replay: "GenericReplay"):
        self.replay = replay

    def __iter__(self):
        return self.replay.dataset()


def inspect_batch_unittest(batch, config):
    # Action to move
    ACTION_MAP = {0: "LEFT", 1: "RIGHT", 2: "UP", 3: "DOWN", 4: "WAIT"}

    # Movement effect: (dx, dy)
    MOVE_EFFECT = {
        "LEFT": (0, -1),
        "RIGHT": (0, +1),
        "UP": (-1, 0),
        "DOWN": (+1, 0),
        "WAIT": (0, 0),
    }

    # write everything to file
    np.set_printoptions(threshold=np.inf)
    file_name = f"batch_{config.task}_{config.env.lwm.disappear}.txt"
    f = open(file_name, "w")
    cprint(f"batch.inspect_write | path={file_name}", "green")

    # check avatar_pos is moving in the correct direction
    def get_action_name(action_vec):
        idx = np.argmax(action_vec)
        return ACTION_MAP[idx]

    for sample_id in tqdm(range(batch["action"].shape[0]), desc="Inspecting batch"):
        sample = {}
        for k, v in batch.items():
            if k != "rng":
                sample[k] = v[sample_id]

        # transfer sample to host
        sample = convert_mets(sample, batch["rng"].devices())

        # do not write numpy array with ...
        # write the full numpy array without truncation
        for k, v in sample.items():
            f.write(f"{k}\n")
            if isinstance(v, np.ndarray):
                for row_id, row in enumerate(v):
                    f.write(f"{row_id} {row}\n")
            else:
                f.write(f"{v}\n")
            f.write("\n")
        f.write("\n")

        avatar_pos = sample["avatar_pos"]
        for i in range(avatar_pos.shape[0] - 1):
            if sample["is_last"][i]:
                if config.env.lwm.disappear:
                    pass
                    # num_alive_entities = [e for e in sample["entity_ids"][i] if e != 0]
                    # is_dead_agent = (
                    #     sample["avatar_ids"][i][0] == 0,
                    #     f"{sample['avatar_ids'][i]=}",
                    # )
                    # assert is_dead_agent or len(num_alive_entities) < 3, (  # type: ignore
                    #     sample["entity_ids"][i],
                    #     sample["avatar_ids"][i],
                    # )
                else:
                    # not disappear
                    num_alive_entities = [e for e in sample["entity_ids"][i] if e != 0]
                    if sample["reward"][i] == -1:
                        if sample["avatar_ids"][i][0] == 16:
                            assert len(num_alive_entities) == 2, (
                                f"{sample['entity_ids'][i]=}"
                            )
                        else:
                            assert len(num_alive_entities) == 3, (
                                f"{sample['entity_ids']=}"
                            )
                    if sample["reward"][i] == 1:
                        assert len(num_alive_entities) == 2, (
                            f"{sample['entity_ids'][i]=}"
                        )
                    assert sample["avatar_ids"][i][0] in [15, 16], (
                        f"{sample['avatar_ids'][i]=}"
                    )

            action_name = get_action_name(sample["action"][i])
            avatar_pos_next = avatar_pos[i][0][:2] + MOVE_EFFECT[action_name]
            # bound check
            avatar_pos_next = np.clip(avatar_pos_next, 0, 9)
            next_is_last = sample["is_last"][i + 1]
            if not next_is_last and not sample["is_last"][i]:
                assert np.all(avatar_pos_next == sample["avatar_pos"][i + 1][0][:2]), (
                    avatar_pos_next,
                    sample["avatar_pos"][i + 1][0][:2],
                )
    cprint("PASS", "green")


tree_map = jax.tree_util.tree_map


def convert_mets(value, devices):
    if len(devices) > 1:
        value = tree_map(lambda x: x[0], value)
    return jax.device_get(value)


def parallel_eval(
    step: "Counter",
    real_env_step: "Counter",
    opt_step: "Counter",
    agent: "JAXAgent",
    replay: "Uniform",
    logger: "Logger",
    timer: "Timer",
    # usage: "Usage",
    args,
    config,
    eval_key,  # eval or test
    env_cache=None,
):
    cprint(f"eval.start | mode={eval_key}")
    metrics = Metrics()
    should_log = Clock(args.log_every)
    dataset = agent.dataset(replay.dataset, args.report.first_bs, 2)  # type: Batcher
    stats = dict(
        last_time=time.time(),
        last_step=int(step),
        batch_entries=0,
        last_real_step=real_env_step.value,
        last_opt_step=opt_step.value,
    )
    event_logger.info("eval.ready | mode={} | env_step={}", eval_key, int(step))

    first_batch = True
    first_report = True
    while True:
        start_batch = time.time()
        if first_batch:
            with long_operation_progress(
                f"{eval_key} first replay batch",
                lambda: _replay_startup_detail(replay),
            ):
                batch = next(dataset)
            first_batch = False
        else:
            batch = next(dataset)

        if should_log():
            if first_report:
                # Report compilation is large and uses the same train devices as
                # the learner. Let the first learner JIT finish instead of
                # compiling train, eval, and test concurrently at startup.
                with long_operation_progress(
                    f"{eval_key} report deferred",
                    "waiting for the first learner update to complete",
                ):
                    _wait_for_first_agent_update(agent)
                with long_operation_progress(
                    f"{eval_key} first report JIT",
                    "tracing/compiling report metrics on train devices",
                ):
                    report = agent.report(
                        batch,
                        step=opt_step if args.opt_step else None,
                    )  # Agent.report
                first_report = False
            else:
                report = agent.report(
                    batch,
                    step=opt_step if args.opt_step else None,
                )  # Agent.report

            # if wandb is running
            if config.use_wandb:
                # if "rollout_reward_kl_loss_scaled_h=1" in report:
                #     draw_rollout_reward_kl_loss_each_step(report, config, name=eval_key)
                # if "rollout_dyn_kl_loss_scaled_h=1" in report:
                #     draw_rollout_dyn_kl_loss_each_step(report, config, name=eval_key)
                if "dyn_loss_mean_t=1" in report:
                    draw_loss_each_step(
                        report, config, loss_name="dyn_loss_mean", name=eval_key
                    )
                if "rollout_reward_loss_scaled_t=1" in report:
                    draw_loss_each_step(
                        report,
                        config,
                        loss_name="rollout_reward_loss_scaled",
                        name=eval_key,
                    )
                if "rollout_dyn_loss_scaled_t=1" in report:
                    draw_loss_each_step(
                        report,
                        config,
                        loss_name="rollout_dyn_loss_scaled",
                        name=eval_key,
                    )

            # if 's1' in config.task:
            #     if 'reward_data_dist' in report:
            #         assert report['reward_data_dist']

            # if "rollout_cont_prob" in report:
            #     wandb.log(
            #         {
            #             f"{eval_key}_rollout_cont_curve": wandb.Image(
            #                 draw_precision_recall_curve(report)
            #             ),
            #         }
            #     )
            # del report["rollout_cont_prob"]
            # del report["rollout_cont_label"]
            # del report["rollout_cont_mask"]

            if args.use_table:
                report.update(make_table_data_text(report, env_cache))

            logger.add(report, prefix=f"{eval_key}/report")
            logger.add({"real_step": real_env_step.value})
            # logger.add(timer.stats(), prefix="timer")
            replay_stats = replay.stats
            logger.add(replay_stats, prefix=f"{eval_key}_replay")

            if replay_stats["insert_wait_frac"] > 0.2:
                cprint(
                    f"replay.wait_pressure | operation=insert | "
                    f"fraction={replay_stats['insert_wait_frac']:.3f}",
                    "red",
                )
            if replay_stats["sample_wait_frac"] > 0.2:
                cprint(
                    f"replay.wait_pressure | operation=sample | "
                    f"fraction={replay_stats['sample_wait_frac']:.3f}",
                    "red",
                )
            # logger.add(usage.stats(), prefix="usage")

            # duration = time.time() - stats["last_time"]
            # actor_fps = (int(step) - stats["last_step"]) / duration
            # learner_fps = stats["batch_entries"] / duration
            # real_fps = (real_env_step.value - stats["last_real_step"]) / duration
            # stats = dict(
            #     last_time=time.time(),
            #     last_step=int(step),
            #     batch_entries=0,
            #     last_real_step=real_env_step.value,
            #     last_opt_step=opt_step.value,
            # )
            logger.write(fps=True)


def parallel_learner(
    env_step: "Counter",  # logger.step
    real_env_step: "Counter",
    opt_step: "Counter",
    agent: "JAXAgent",
    train_replay: "GenericReplay",
    logger: "Logger",
    # timer: "Timer",
    # usage: "Usage",
    checkpoint: "Checkpoint",
    args,
    id2sent=None,
    # train_ratio=None,
    config=None,
):
    cprint("learner.start")
    metrics = Metrics()
    should_log = Clock(args.log_every)
    should_log_progress = Clock(getattr(args, "learner_log_every", 60))
    should_save = Clock(args.save_every)
    not_saved = Once()
    train_dataset = agent.dataset(train_replay.dataset)  # type: Batcher
    start_batch = time.time()
    # train_dataset = agent.torch_dataloader(Dataset(train_replay))
    state = None
    stats = dict(
        last_time=time.time(),
        last_step=int(env_step),
        batch_entries=0,
        last_real_step=real_env_step.value,
        last_opt_step=opt_step.value,
    )
    event_logger.info("learner.ready | env_step={}", int(env_step))
    startup_batch_size = getattr(config, "batch_size", None)
    if startup_batch_size is None:
        startup_batch_size = getattr(agent, "batch_size", "unknown")
    with long_operation_progress(
        "learner first replay batch",
        lambda: _replay_startup_detail(train_replay),
    ):
        batch = next(train_dataset)
    recent_mean_seq_sample, recent_mean_priorities = consume_replay_sample_stats(batch)
    batch_time = time.time() - start_batch
    cprint("learner.first_batch", "yellow")
    # The replay has now proven that it contains a complete unique learner
    # batch. Persist that initial fill immediately so the next compatible run
    # can restore it while this run continues compiling/training.
    train_replay.save()
    cprint(
        f"replay.prefill_snapshot | state=started | buffer_size={len(train_replay)}",
        "green",
    )
    total_size = sum(np.prod(v.shape) for v in batch.values())
    for k, v in batch.items():
        cprint(
            f"learner.batch_field | name={k} | shape={v.shape} | "
            f"fraction={np.prod(v.shape) / total_size:.2%}"
        )

    cnt_grad = 0
    batch_data_ids = []
    batch_data_priorities = []
    replay_starved = False

    # inspect the first batch
    # inspect_batch_unittest(batch, config)

    while True:
        # cprint(f"[learner] Got batch in {time.time() - start_batch}s", "green")
        opt_step.increment()  # optimize_step
        # state: prev_latent (deter, stoch, logit), prev_action
        start_train = time.time()

        assert config is not None, "config is None"
        bw_dict = _train_reward_weights(train_replay, config)

        if cnt_grad == 0:
            startup_batch_length = getattr(config, "batch_length", None)
            if startup_batch_length is None:
                startup_batch_length = getattr(agent, "batch_length", "unknown")
            with long_operation_progress(
                "learner first train JIT",
                f"batch_size={startup_batch_size}, batch_length={startup_batch_length}",
            ):
                outs, _, mets = agent.train(
                    batch,
                    state=None,
                    step=opt_step.value if args.opt_step else None,
                    imbalanced_reward_weights=bw_dict,
                )
        else:
            outs, _, mets = agent.train(
                batch,
                state=None,
                step=opt_step.value if args.opt_step else None,
                imbalanced_reward_weights=bw_dict,
            )
        train_time = time.time() - start_train
        metrics.add(mets)
        stats["batch_entries"] += batch["is_first"].size

        # update the buffer based on priority
        update_time = 0
        num_duplicated_keys = 0
        uuid_keys = None
        if "sample_id" in mets:
            uuid_keys = convert2uuid(mets["sample_id"])
            num_duplicated_keys = len(uuid_keys) - len(set(uuid_keys))
            batch_data_ids.extend(uuid_keys)

        if hasattr(train_replay.sampler, "update_priorities"):
            assert isinstance(train_replay.sampler, PrioritizedSampler)
            if uuid_keys is not None:
                batch_data_priorities.extend(
                    [train_replay.sampler.key2priority[key] for key in uuid_keys]
                )

            if "sample_id" in mets:
                assert PRIORITY_KEY in mets, f"{PRIORITY_KEY} not in mets"

            if "sup_loss_per_batch" in mets:
                # assert not nan
                if np.isnan(mets["sup_loss_per_batch"]).any():
                    cprint(
                        f"learner.nan_detected | metric=sup_loss_per_batch | "
                        f"value={mets['sup_loss_per_batch']}",
                        "red",
                    )
                    raise ValueError("NaN in sup_loss_per_batch")

            if "rollout_loss_per_batch" in mets:
                # assert not nan
                if np.isnan(mets["rollout_loss_per_batch"]).any():
                    cprint(
                        f"learner.nan_detected | metric=rollout_loss_per_batch | "
                        f"value={mets['rollout_loss_per_batch']}",
                        "red",
                    )
                    raise ValueError("NaN in rollout_loss_per_batch")

            if PRIORITY_KEY in mets:
                start = time.time()
                # if config.replay.balanced_weight_priority:
                #     assert bw_dict is not None
                #     # bs,
                #     batch_weights = np.vectorize(bw_dict.get)(
                #         mets["reward_indicator_dist"]
                #     )
                #     if np.isnan(mets[PRIORITY_KEY]).any():
                #         cprint(
                #             f"NaN detected in mets[PRIORITY_KEY]: {mets[PRIORITY_KEY]}",
                #             "red",
                #         )
                #     if np.isnan(batch_weights).any():
                #         cprint(f"NaN detected in batch_weights: {batch_weights}", "red")
                #     model_loss = mets[PRIORITY_KEY] * batch_weights
                # else:
                model_loss = mets[PRIORITY_KEY]
                if np.isnan(model_loss).any():
                    cprint(
                        f"learner.nan_detected | metric=model_loss | "
                        f"weighted=false | value={model_loss}",
                        "red",
                    )
                assert not np.isnan(model_loss).any(), model_loss
                assert uuid_keys is not None, uuid_keys

                train_replay.sampler.update_priorities(
                    model_loss,
                    uuid_keys=uuid_keys,
                )

                # if hasattr(train_replay, "sampler_pos"):
                if config.replay.imbalance == "upsample_pos":
                    assert isinstance(train_replay.sampler_pos, PrioritizedSampler)
                    assert uuid_keys is not None
                    train_replay.sampler_pos.update_priorities(
                        # mets["sample_id"],
                        mets[PRIORITY_KEY],
                        uuid_keys=uuid_keys,
                    )

                update_time = time.time() - start

        replay_avail = train_replay.limiter.avail_ratio
        replay_starved = _warn_replay_starvation(
            replay_avail,
            replay_starved,
            int(opt_step),
        )
        progress_log_now = should_log_progress()
        if progress_log_now:
            messenger = str(config.logdir)
            messenger_prefix = "logdir/messenger/"
            if messenger.startswith(messenger_prefix):
                messenger = messenger[len(messenger_prefix) :]
            cprint(
                f"learner.step | opt_step={int(opt_step)} | "
                f"replay_avail={replay_avail:.2%} | "
                f"batch_time={batch_time:.3f}s | train_time={train_time:.3f}s | "
                f"priority_update_time={update_time:.3f}s | run={messenger}"
            )
            cprint(
                f"learner.replay | pos_stream_rate={train_replay.pos_stream_rate:g} | "
                f"pos_eps_rate={train_replay.pos_eps_rate:g} | "
                f"mean_sample_count={recent_mean_seq_sample} | "
                f"mean_priority={recent_mean_priorities} | "
                f"duplicate_keys={num_duplicated_keys}"
            )
        metrics.add(
            {
                "batch_time": batch_time,
                "train_time": train_time,
                "update_time": update_time,
            },
            prefix="timer",
        )
        cnt_grad += 1

        if should_log():
            # if config.use_wandb:
            #     draw_priority_replay_info(
            #         train_replay, config, batch_data_ids, batch_data_priorities
            #     )

            train_metrics = metrics.result()
            if config.use_wandb:
                # if "rollout_reward_kl_loss_scaled_h=1" in train_metrics:
                #     draw_rollout_reward_kl_loss_each_step(
                #         train_metrics, config, name="train"
                #     )
                # if "rollout_dyn_kl_loss_scaled_h=1" in train_metrics:
                #     draw_rollout_dyn_kl_loss_each_step(
                #         train_metrics, config, name="train"
                #     )
                if "dyn_loss_mean_t=1" in train_metrics:
                    draw_loss_each_step(
                        train_metrics, config, loss_name="dyn_loss_mean", name="train"
                    )
                if "rollout_reward_loss_scaled_t=1" in train_metrics:
                    draw_loss_each_step(
                        train_metrics,
                        config,
                        loss_name="rollout_reward_loss_scaled",
                        name="train",
                    )
                if "rollout_dyn_loss_scaled_t=1" in train_metrics:
                    draw_loss_each_step(
                        train_metrics,
                        config,
                        loss_name="rollout_dyn_loss_scaled",
                        name="train",
                    )

            if args.report.train:
                report = agent.report(batch, step=opt_step if args.opt_step else None)
                report = {k: v for k, v in report.items() if k not in train_metrics}

                # if config.use_wandb:
                #     if "rollout_cont_prob" in report:
                #         wandb.log(
                #             {
                #                 "train_rollout_cont_curve": wandb.Image(
                #                     draw_precision_recall_curve(report)
                #                 ),
                #             }
                #         )
                # del report["rollout_cont_prob"]
                # del report["rollout_cont_label"]
                # del report["rollout_cont_mask"]

                if args.use_table:
                    report.update(make_table_data_text(report, id2sent))
                logger.add(report, prefix="train/report")

            logger.add({"real_step": real_env_step.value})
            logger.add(train_metrics, prefix="train")

            if (
                isinstance(train_replay.sampler, PrioritizedSampler)
                and PRIORITY_KEY in mets
            ):
                assert isinstance(train_replay.sampler, PrioritizedSampler)
                assert uuid_keys is not None
                priorities_dist = [
                    train_replay.sampler.key2priority[key]
                    for key in uuid_keys
                    if train_replay.sampler.has_key(key)
                ]
                # if hasattr(train_replay, "sampler_pos"):
                if config.replay.imbalance == "upsample_pos":
                    assert isinstance(train_replay.sampler_pos, PrioritizedSampler)
                    priorities_dist += [
                        train_replay.sampler_pos.key2priority[key]
                        for key in uuid_keys
                        if train_replay.sampler_pos.has_key(key)
                    ]

                # batch_data_priorities = priorities_dist
                priorities_dist = np.array(priorities_dist).reshape(-1)
                priorities_dist_loss = priorities_dist[priorities_dist < 50]

                priority_stats = {
                    "mean_priorities": np.mean(priorities_dist),
                    "mean_sample": recent_mean_seq_sample,
                    "priorities_dist": priorities_dist,
                    "priorities_dist_loss": priorities_dist_loss,
                    # "model_loss": model_loss.reshape(-1),
                }
            else:
                priority_stats = {}

            logger.add(
                {
                    **train_replay.stats,
                    **priority_stats,
                    "duplicated_keys": num_duplicated_keys,
                },
                prefix="replay",
            )

            duration = time.time() - stats["last_time"]
            actor_fps = (int(env_step) - stats["last_step"]) / duration
            learner_fps = stats["batch_entries"] / duration
            real_fps = (real_env_step.value - stats["last_real_step"]) / duration
            logger.add(
                {
                    "actor_fps": actor_fps,
                    "learner_fps": learner_fps,
                    "train_ratio": learner_fps / actor_fps if actor_fps else np.inf,
                    "real_fps": real_fps,
                    "ops": (opt_step.value - stats["last_opt_step"]) / duration,
                },
                prefix="parallel",
            )

            stats = dict(
                last_time=time.time(),
                last_step=int(env_step),
                batch_entries=0,
                last_real_step=real_env_step.value,
                last_opt_step=opt_step.value,
            )
            logger.write(fps=True)

        if not config.run.debug and should_save():
            path = pathlib.Path(args.logdir) / "checkpoint_1.ckpt"  # final
            checkpoint.save(path)
            # Rotate and persist the raw replay streams alongside each model
            # checkpoint. This makes the next compatible S1 run start from a
            # populated replay instead of waiting for all actors to refill it.
            train_replay.save()

        start_batch = time.time()
        batch = next(train_dataset)
        recent_mean_seq_sample, recent_mean_priorities = consume_replay_sample_stats(
            batch
        )
        batch_time = time.time() - start_batch


def draw_rollout_reward_kl_loss_each_step(metrics, config, name="train"):
    x = range(config.imag_horizon)
    y_kl_loss = [
        metrics[f"rollout_reward_kl_loss_scaled_h={i + 1}"]
        for i in range(config.imag_horizon)
    ]
    y_kl_count = [
        metrics[f"rollout_reward_kl_count_h={i + 1}"]
        for i in range(config.imag_horizon)
    ]
    # plot line chart with dual y-axes
    fig = plt.figure()
    ax1 = fig.add_subplot(111)

    color1 = "tab:blue"
    ax1.set_xlabel("step")
    ax1.set_ylabel("rollout_reward_kl_loss", color=color1)
    ax1.plot(x, y_kl_loss, color=color1)
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.grid(True)

    ax2 = ax1.twinx()
    color2 = "tab:red"
    ax2.set_ylabel("rollout_reward_kl_count", color=color2)
    ax2.plot(x, y_kl_count, color=color2)
    ax2.tick_params(axis="y", labelcolor=color2)

    plt.title("Rollout Reward KL Loss and KL Count per Step")
    wandb.log({f"{name}_rollout_reward_kl_metrics_fig": wandb.Image(fig)})


def draw_rollout_dyn_kl_loss_each_step(metrics, config, name="train"):
    x = range(config.imag_horizon)
    y_kl_loss = [
        metrics[f"rollout_dyn_kl_loss_scaled_h={i + 1}"]
        for i in range(config.imag_horizon)
    ]
    y_kl_count = [
        metrics[f"rollout_dyn_kl_count_h={i + 1}"] for i in range(config.imag_horizon)
    ]
    # plot line chart with dual y-axes
    fig = plt.figure()
    ax1 = fig.add_subplot(111)

    color1 = "tab:blue"
    ax1.set_xlabel("step")
    ax1.set_ylabel("rollout_dyn_kl_loss", color=color1)
    ax1.plot(x, y_kl_loss, color=color1)
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.grid(True)

    ax2 = ax1.twinx()
    color2 = "tab:red"
    ax2.set_ylabel("rollout_dyn_kl_count", color=color2)
    ax2.plot(x, y_kl_count, color=color2)
    ax2.tick_params(axis="y", labelcolor=color2)

    plt.title("Rollout Dyn KL Loss and KL Count per Step")
    wandb.log({f"{name}_rollout_dyn_kl_metrics_fig": wandb.Image(fig)})


def draw_loss_each_step(metrics, config, loss_name="dyn_loss_mean", name="train"):
    x = range(config.imag_horizon)
    y_loss = [metrics[f"{loss_name}_t={i + 1}"] for i in range(config.imag_horizon)]
    # plot line chart with dual y-axes
    fig = plt.figure()
    ax1 = fig.add_subplot(111)

    color1 = "tab:blue"
    ax1.set_xlabel("step")
    ax1.set_ylabel(loss_name, color=color1)
    ax1.plot(x, y_loss, color=color1)
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.grid(True)

    plt.title(f"{loss_name} per Step")
    wandb.log({f"{name}_{loss_name}_metrics_fig": wandb.Image(fig)})


def draw_priority_replay_info(
    train_replay,
    config,
    batch_data_ids,
    batch_data_priorities,
):
    fig = draw_dist_whole_dataset(train_replay.num_samples_whole_dataset, config.logdir)
    wandb.log({"sample_data_fig": wandb.Image(fig)})
    if isinstance(train_replay.sampler, PrioritizedSampler):
        fig_p = draw_dist_whole_dataset(train_replay.priority_dataset, config.logdir)
        wandb.log({"priority_data_fig": wandb.Image(fig_p)})
        fig_loss = draw_dist_whole_dataset(
            train_replay.priority_loss_dataset, config.logdir
        )
        wandb.log({"priority_loss_data_fig": wandb.Image(fig_loss)})
        fig_sample = draw_hist_whole_dataset(
            batch_data_ids, train_replay.table, config.logdir
        )
        wandb.log({"sample_batch_fig": wandb.Image(fig_sample)})
        batch_data_ids = []
        fig_priority = draw_hist_priority(batch_data_priorities)
        wandb.log({"batch_priorities_fig": wandb.Image(fig_priority)})
        batch_data_priorities = []


def need_longer_time_to_load(args):
    return args.resume != "" or args.debug or args.script == "parallel_eval"


def dummy_data(spaces, batch_dims):
    # TODO: Get rid of this function by adding initial_policy_state() and
    # initial_train_state() to the agent API.
    assert batch_dims[0] != 0, batch_dims
    spaces = list(spaces.items())
    data = {k: np.zeros(v.shape, v.dtype) for k, v in spaces}
    for dim in reversed(batch_dims):
        data = {k: np.repeat(v[None], dim, axis=0) for k, v in data.items()}
    return data


def parallel_finetune_wm(
    agent: "JAXAgent",
    train_replay: "Uniform | PrioritizedReplay | ReplayEps",
    logger: "Logger",
    make_train_env,
    args,
    env_cache,
    make_eval_env=None,
    make_test_env=None,
    eval_replay: "Uniform|None" = None,
    test_replay=None,
    config: "Config|None" = None,
):
    logdir = Path(args.logdir)
    checkpoint = Checkpoint(logdir / "checkpoint_1.ckpt")
    env_step = logger.step
    real_env_step = Counter()
    opt_step = Counter()
    checkpoint.step = env_step
    checkpoint.real_step = real_env_step
    checkpoint.agent = agent
    checkpoint.opt_step = opt_step
    checkpoint.best_eval_win_rate = Counter()
    checkpoint.best_eval_agent = None

    if args.from_checkpoint != "":
        checkpoint.load(args.from_checkpoint, skip_key=args.skip_key, strict=True)
    if args.load_checkpoint:
        checkpoint.load_or_save()

    # cprint(f"Resume: {env_step=}, {real_env_step=}", "red")
    # if wandb is running
    if wandb.run is not None:
        cprint(f"run.wandb_resume | step={wandb.run.step}")
        env_step.load(wandb.run.step)
        checkpoint.step = env_step

    timer = Timer()
    timer.wrap("agent", agent, ["policy", "train", "report", "save"])
    timer.wrap("replay", train_replay, ["add", "save"])
    timer.wrap("logger", logger, ["write"])
    # usage = Usage(args.trace_malloc)

    workers = []
    event_logger.info("run.workers_start")
    global start_parallel
    start_parallel = time.time()
    assert config is not None, "config is None"
    train_env_ids: List[int] = list(range(config.envs.amount))  # type: ignore
    worker_addr2type = {get_env_address(i): "train" for i in train_env_ids}
    assert env_cache is not None
    id2sent = id2sent_from_env_cache(env_cache)

    if make_eval_env is not None:
        assert args.script in ["parallel_train_eval", "parallel_train_eval_test"]
        assert isinstance(config.num_eval_envs, int)
        eval_env_ids = list(
            range(len(train_env_ids), len(train_env_ids) + config.num_eval_envs)
        )
        worker_addr2type.update({get_env_address(i): "eval" for i in eval_env_ids})

        workers += _spawn_env_workers(
            eval_env_ids,
            make_eval_env,
            args,
            worker_addr2type,
            timer,
            "parallel_env_eval",
            thread_env_id=1,
        )

        if "test" in args.script:
            event_logger.debug("env.group_init | mode=test")
            assert make_test_env is not None, "make_test_env is None"
            test_env_ids = list(
                range(
                    len(worker_addr2type), len(worker_addr2type) + config.num_eval_envs
                )
            )
            worker_addr2type.update({get_env_address(i): "test" for i in test_env_ids})

            workers += _spawn_env_workers(
                test_env_ids,
                make_test_env,
                args,
                worker_addr2type,
                timer,
                "parallel_env_test",
                thread_env_id=1,
            )

    # TRAIN ENVS
    workers += _spawn_env_workers(
        train_env_ids,
        make_train_env,
        args,
        worker_addr2type,
        timer,
        "parallel_env",
        thread_env_id=0,
    )

    workers.append(
        distr.Thread(
            parallel_actor,
            env_step,
            real_env_step,
            opt_step,
            agent,
            train_replay,
            logger,
            # timer,
            args,
            worker_addr2type,
            checkpoint,
            eval_replay,
            test_replay,
        )
    )

    workers.append(
        distr.Thread(
            parallel_learner,
            env_step,
            real_env_step,
            opt_step,
            agent,
            train_replay,
            logger,
            # timer,
            # usage,
            checkpoint,
            args,
            id2sent,
            config,
        )
    )

    if make_eval_env is not None:
        workers.append(
            distr.Thread(
                parallel_eval,
                env_step,
                real_env_step,
                opt_step,
                agent,
                eval_replay,
                logger,
                timer,
                args,
                "eval",
                id2sent,
            )
        )

    if make_test_env is not None:
        workers.append(
            distr.Thread(
                parallel_eval,
                env_step,
                real_env_step,
                opt_step,
                agent,
                test_replay,
                logger,
                timer,
                args,
                "test",
                id2sent,
            )
        )

    distr.run(workers)
