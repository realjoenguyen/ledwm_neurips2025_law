from multiprocessing import Lock, Value
import re
import sys
import time
from collections import OrderedDict, defaultdict
import wandb
from ledwm.embodied.core.uuid import uuid
from typing import TYPE_CHECKING, Dict
import jax
from termcolor import cprint
from ledwm.embodied.core.base import Agent
from ledwm.embodied.replay.generic import (
    GenericReplay,
    consume_replay_sample_stats,
    convert2uuid,
)

if TYPE_CHECKING:
    from ledwm.embodied.core.logger import Logger
    from ledwm.embodied.replay.replays import Uniform
    from ledwm.jaxagent import JAXAgent
    from ledwm.embodied.core.base import Agent
    from ledwm.embodied.replay.limiters import SamplesPerInsert

from ledwm.embodied.replay.Prioritized import PrioritizedReplay, PrioritizedSampler
from ledwm.embodied.run.smoothing import ReplayEps
import numpy as np
from ledwm.embodied.core.WandBOutput import make_table_data_text
from ledwm.embodied.core.checkpoint import Checkpoint
from ledwm.embodied.core.counter import Counter
from ledwm.embodied.core.path import Path
from ledwm.embodied.core.timer import Timer
from ledwm.embodied.core.metrics import Metrics
from ledwm.embodied.core.when import Clock
from ledwm.embodied.core.basics import treemap, convert
from ledwm.embodied.core import distr
from ledwm.embodied.core.batcher import Batcher


def id2sent_from_env_cache(env_cache):
    # sent2id: from sent to id
    # return id2sent: from id to sent
    sent2id = env_cache["sent_ids"]
    id2sent = {v: k for k, v in sent2id.items()}
    return id2sent


def parallel_finetune(
    agent: "JAXAgent",
    replay: "Uniform | PrioritizedReplay | ReplayEps",
    logger: "Logger",
    make_env,
    args,
    env_cache,
    config=None,
):
    logdir = Path(args.logdir)
    checkpoint = Checkpoint(logdir / "checkpoint.ckpt")
    env_step = logger.step
    real_env_step = Counter()
    opt_step = Counter()

    # CHECKPOINT
    checkpoint.step = env_step
    checkpoint.real_step = real_env_step
    checkpoint.agent = agent
    checkpoint.opt_step = opt_step
    if args.from_checkpoint != "":
        checkpoint.load(args.from_checkpoint, skip_key=args.skip_key, strict=True)
    if args.load_checkpoint:
        checkpoint.load_or_save()
    cprint(f"Resume: {env_step=}, {real_env_step=}", "red")
    if wandb.run is not None:
        cprint(f"Wandb step: {wandb.run.step}", "red")
        env_step.load(wandb.run.step)
        checkpoint.step = env_step

    timer = Timer()
    timer.wrap("agent", agent, ["policy", "train", "report", "save"])
    timer.wrap("replay", replay, ["add", "save"])
    timer.wrap("logger", logger, ["write"])
    # usage = embodied.Usage(args.trace_malloc)

    workers = []
    print("start all threads")
    global start_parallel
    start_parallel = time.time()
    env_ids = list(range(config.envs.amount))
    worker_addr2type = {get_env_address(i): "train" for i in env_ids}
    assert env_cache is not None
    id2sent = id2sent_from_env_cache(env_cache)

    if len(env_ids) == 1:
        workers.append(
            distr.Thread(parallel_env, 0, make_env, args, worker_addr2type, timer)
        )
    else:
        for i in env_ids:
            worker = distr.Process(
                parallel_env,
                i,
                make_env,
                args,
                worker_addr2type,
                name=f"parallel_env_{i}",
            )
            worker.start()
            workers.append(worker)
    # usage.processes("envs", workers)  # envs_count

    workers.append(
        distr.Thread(
            parallel_actor,
            env_step,
            real_env_step,
            opt_step,
            agent,
            replay,
            logger,
            args,
            worker_addr2type,
            checkpoint,
        )
    )

    workers.append(
        distr.Thread(
            parallel_learner,
            env_step,
            real_env_step,
            opt_step,
            agent,
            replay,
            logger,
            # timer,
            # usage,
            checkpoint,
            args,
            id2sent,
            config,
        )
    )

    distr.run(workers)


def get_env_address(val: int):
    return val.to_bytes(16, "big").hex()


def parallel_actor(
    env_step: "embodied.Counter",  # logger.step
    real_env_step: "embodied.Counter",
    opt_step: "embodied.Counter",
    agent: "JAXAgent",
    train_replay: "ReplayEps",
    logger: "Logger",
    # timer: "Timer",
    args,
    worker_addr2type: Dict[int, bytes],
    checkpoint: "Checkpoint",
):
    cprint("[actor] start parallel_actor", "green")
    metrics = Metrics()
    # to store all steps in one eps - for each env
    scalars_eps = defaultdict(lambda: defaultdict(list))

    should_log_video = args.log_video
    cprint(
        f"[actor] should_log_video = {should_log_video}",
        "green" if should_log_video else "yellow",
    )
    if should_log_video:
        videos = defaultdict(lambda: defaultdict(list))  # addr2video
    should_log = Clock(args.log_every)
    act = None
    train_win_rates = []

    def callback(obs, env_addrs):
        obs_steps = np.zeros((args.actor_batch), dtype=np.int32)
        for i, addr in enumerate(env_addrs):
            if obs["is_first"][i]:
                abandoned = not dones.get(addr, True)
                metrics.scalar("parallel/episode_abandoned", int(abandoned), agg="sum")
            dones[addr] = obs["is_last"][i]
            # -1 for eval env, no unimix for eval policy
            obs_steps[i] = opt_step.value if worker_addr2type[addr] == "train" else -1

        states = [allstates[a] for a in env_addrs]
        states = treemap(lambda *xs: list(xs), *states)

        # take action from ALL? policy
        # act, states = agent.policy(
        #     obs, states, step=obs_steps if args.opt_step else None
        # )  # (env, ...)
        act, states, info = agent.policy(
            obs, states, step=obs_steps, mode="train"
        )  # (env, ...)

        # make sure act does not have nan
        assert not np.any(np.isnan(act["action"])), act["action"]

        act["reset"] = obs["is_last"].copy()
        for i, addr in enumerate(env_addrs):
            allstates[addr] = treemap(lambda x: x[i], states)

        env_step.increment(args.actor_batch)  # += actor_batch == env
        real_env_step.increment(
            (~obs["is_read_step"]).sum() if "is_read_step" in obs else args.actor_batch
        )
        metrics.scalar("parallel/ep_states", len(allstates))

        trans = {**obs, **act}
        now = time.time()

        for i, addr in enumerate(env_addrs):
            tran = {k: v[i].copy() for k, v in trans.items()}

            if worker_addr2type[addr] == "train":
                train_replay.add(tran.copy(), worker=addr)  # Blocks when rate limited.
                # step_before[addr] = tran.copy()

            elif worker_addr2type[addr] == "eval":
                assert eval_replay is not None
                eval_replay.add(tran.copy(), worker=addr, training=False)
                # step_before[addr] = tran.copy()

            elif worker_addr2type[addr] == "test":
                assert test_replay is not None
                test_replay.add(tran.copy(), worker=addr, training=False)
            else:
                raise ValueError(f"Unknown env type: {worker_addr2type[addr]}")

            # video record
            if tran["is_first"]:
                scalars_eps.pop(addr, None)
                if should_log_video:
                    videos.pop(addr, None)
                    vidstreams.pop(addr, None)

            [scalars_eps[addr][k].append(v) for k, v in tran.items() if v.size == 1]
            if should_log_video:
                if addr in vidstreams or len(vidstreams) < args.log_video_streams:
                    vidstreams[addr] = now
                    [
                        videos[addr][k].append(tran[k])
                        for k in args.log_keys_video
                        if k != ""
                    ]

        if should_log_video:
            for addr, last_add in list(vidstreams.items()):
                if now - last_add > args.log_video_timeout:
                    print(
                        f"Dropping video stream due to timeout ({now - last_add:.1f}s)."
                    )
                    del vidstreams[addr]
                    del videos[addr]

        for i, addr in enumerate(env_addrs):
            if not trans["is_last"][i]:
                continue

            # MUST BE trans['is_last'] for this env
            ep_addr = scalars_eps.pop(addr)
            if should_log_video:
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
                # if score is not nan
                if not np.isnan(score):
                    train_win_rates.append(score > 0)
                if len(train_win_rates) > args.eval_eps:
                    cprint(f"TRAIN Win rate: {np.mean(train_win_rates)}", "green")
                    logger.add({"win_rate": np.mean(train_win_rates)}, prefix="train")
                    train_win_rates.clear()

            elif worker_addr2type[addr] == "eval":
                score = sum(ep_addr["reward"])
                # if score is not nan
                if not np.isnan(score):
                    eval_win_rates.append(score > 0)
                if len(eval_win_rates) > args.eval_eps:
                    # assert win_rates doesn't have NaN for all elements
                    # assert not all(np.isnan(win_rates))
                    win_rate = np.mean(eval_win_rates)
                    cprint(f"EVAL Win rate: {win_rate}", "green")
                    logger.add({"win_rate": np.mean(eval_win_rates)}, prefix="eval")
                    eval_win_rates.clear()
                    # if win_rate > checkpoint.best_dev_win_rate:
                    #     checkpoint.best_dev_win_rate = win_rate
                    #     checkpoint.best_dev_agent = agent
                    #     checkpoint.save()

            elif worker_addr2type[addr] == "test":
                score = sum(ep_addr["reward"])
                if not np.isnan(score):
                    test_win_rates.append(score > 0)
                if len(test_win_rates) > args.eval_eps:
                    cprint(f"TEST Win rate: {np.mean(test_win_rates)}", "green")
                    logger.add({"win_rate": np.mean(test_win_rates)}, prefix="test")
                    test_win_rates.clear()

            if "is_read_step" in ep_addr:
                raise NotImplementedError("is_read_step")

            if worker_addr2type[addr] == "train":
                logger.add({"real_step": real_env_step.value})
                stats = {}
                if should_log_video:
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
                if should_log_video:
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
                if should_log_video:
                    for key in args.log_keys_video:
                        if key in ep_addr:
                            tag = (
                                f"test/win/policy_{key}"
                                if ep_addr["reward"][-1] == 1
                                else f"test/lose/policy_{key}"
                            )
                            stats[tag] = ep_addr[key]
                metrics.add(stats, prefix="stats")

        if should_log():
            logger.add(metrics.result())

        return act

    # state = None
    cprint("[actor] init policy with dummy data", "green")
    # outs, state. Get initial as policy state
    _, initial, info = agent.policy(
        dummy_data(agent.agent.obs_space, (args.actor_batch,)),
        step=opt_step.value if args.opt_step else None,
        mode="train",
    )

    initial = treemap(lambda x: x[0], initial)
    allstates = defaultdict(lambda: initial)
    nonzeros = set()
    vidstreams = {}
    dones = {}

    server = distr.Server(
        callback, args.actor_port, args.ipv6, args.actor_batch, args.actor_threads
    )
    # timer.wrap("server", server, ["_step", "_work"])
    server.run()


# def get_step_for_envs()
from torch.utils.data import IterableDataset, DataLoader


class Dataset(IterableDataset):
    def __init__(self, replay: "GenericReplay"):
        self.replay = replay

    def __iter__(self):
        return self.replay.dataset()


def parallel_eval(
    step: "embodied.Counter",
    real_env_step: "embodied.Counter",
    opt_step: "embodied.Counter",
    agent: "JAXAgent",
    replay: "Uniform",
    logger: "Logger",
    timer: "Timer",
    # usage: "embodied.Usage",
    args,
    eval_key,  # eval or test
    env_cache=None,
):
    cprint("[eval] start parallel_eval", "green")

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
    print(f"[eval] Training at {step=}")

    while True:
        start_batch = time.time()
        batch = next(dataset)

        if should_log():
            report = agent.report(
                batch,
                step=opt_step if args.opt_step else None,
            )  # Agent.report

            if args.use_table:
                report.update(make_table_data_text(report, env_cache))

            logger.add(report, prefix=f"{eval_key}/report")
            logger.add({"real_step": real_env_step.value})
            # logger.add(timer.stats(), prefix="timer")
            logger.add(replay.stats, prefix=f"{eval_key}_replay")

            if replay.stats["insert_wait_frac"] > 0.2:
                cprint(f"Insert wait frac: {replay.stats['insert_wait_frac']}", "red")
            if replay.stats["sample_wait_frac"] > 0.2:
                cprint(f"Sample wait frac: {replay.stats['sample_wait_frac']}", "red")
            # logger.add(usage.stats(), prefix="usage")

            duration = time.time() - stats["last_time"]
            actor_fps = (int(step) - stats["last_step"]) / duration
            learner_fps = stats["batch_entries"] / duration
            real_fps = (real_env_step.value - stats["last_real_step"]) / duration
            stats = dict(
                last_time=time.time(),
                last_step=int(step),
                batch_entries=0,
                last_real_step=real_env_step.value,
                last_opt_step=opt_step.value,
            )
            logger.write(fps=True)


def parallel_learner(
    step: "embodied.Counter",
    real_env_step: "embodied.Counter",
    opt_step: "embodied.Counter",
    agent: "JAXAgent",
    replay: "GenericReplay",
    logger: "Logger",
    checkpoint: "Checkpoint",
    args,
    id2sent=None,
    config=None,
):
    cprint("[learner] start parallel_learner", "green")

    metrics = Metrics()
    should_log = Clock(args.log_every)
    should_save = Clock(args.save_every)

    train_dataset = agent.dataset(replay.dataset)  # type: Batcher
    # train_dataset = agent.torch_dataloader(Dataset(train_replay))
    state = None
    stats = dict(
        last_time=time.time(),
        last_step=int(step),
        batch_entries=0,
        last_real_step=real_env_step.value,
        last_opt_step=opt_step.value,
    )
    print(f"[learner] Training at {step=}")
    batch = next(train_dataset)
    consume_replay_sample_stats(batch)
    total_size = sum(np.prod(v.shape) for v in batch.values())
    for k, v in batch.items():
        print(k, v.shape, np.prod(v.shape) / total_size)

    while True:
        start_batch = time.time()
        batch = next(train_dataset)
        recent_mean_seq_sample, recent_mean_priorities = consume_replay_sample_stats(
            batch
        )
        batch_time = time.time() - start_batch

        # cprint(f"[learner] Got batch in {time.time() - start_batch}s", "green")
        opt_step.increment()  # optimize_step
        # state: prev_latent (deter, stoch, logit), prev_action
        start_train = time.time()

        # if config.replay.imbalance == "balanced_weight":
        #     imbalanced_reward_weights = OrderedDict(
        #         {
        #             k: replay.reward_balanced_weight(k)
        #             for k in replay.imbalanced_rewards
        #         }
        #     )
        # else:
        #     imbalanced_reward_weights = None

        outs, state, mets = agent.train(
            batch,
            state,
            step=opt_step.value if args.opt_step else None,
            imbalanced_reward_weights=None,
        )  # type: Agent.train
        train_time = time.time() - start_train

        metrics.add(mets)
        stats["batch_entries"] += batch["is_first"].size

        # update the buffer based on priority
        update_time = 0
        num_duplicated_keys = 0
        if "sample_id" in mets:
            uuid_keys = convert2uuid(mets["sample_id"])
            num_duplicated_keys = len(uuid_keys) - len(set(uuid_keys))

        if hasattr(replay.sampler, "update_priorities"):
            assert isinstance(replay.sampler, PrioritizedSampler)
            if "sample_id" in mets:
                assert PRIORITY_KEY in mets, f"{PRIORITY_KEY} not in mets"

            if PRIORITY_KEY in mets:
                start = time.time()

                # metrics.add(
                #     {"priorities_dist": priorities_dist},
                #     prefix="replay",
                # )

                replay.sampler.update_priorities(
                    # mets["sample_id"],
                    mets[PRIORITY_KEY],
                    uuid_keys=uuid_keys,
                )
                # if hasattr(train_replay, "sampler_pos"):
                if config.replay.imbalance == "upsample_pos":
                    replay.sampler_pos.update_priorities(
                        # mets["sample_id"],
                        mets[PRIORITY_KEY],
                        uuid_keys=uuid_keys,
                    )

                update_time = time.time() - start

        current_time = time.time()
        # print in form of date, hour, minute, second
        current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(current_time))

        cprint(
            f"[learner]: avail {replay.limiter.avail_ratio},  name = {config.logdir}, {batch_time=}, {train_time=}, {update_time=}"
        )
        cprint(f"{replay.pos_stream_rate=}, {replay.pos_eps_rate=}")
        cprint(
            f"{recent_mean_seq_sample=}, {recent_mean_priorities=}, {num_duplicated_keys=}, {opt_step=}, {current_time=}",
        )

        metrics.add(
            {
                "batch_time": batch_time,
                "train_time": train_time,
                "update_time": update_time,
            },
            prefix="timer",
        )

        if should_log():
            train_metrics = metrics.result()
            if args.report.train:
                report = agent.report(
                    batch, step=opt_step if args.opt_step else None
                )  # agent.report
                report = {k: v for k, v in report.items() if k not in train_metrics}
                if args.use_table:
                    report.update(make_table_data_text(report, id2sent))

                logger.add(report, prefix="train/report")

            logger.add({"real_step": real_env_step.value})
            logger.add(train_metrics, prefix="train")
            # logger.add(timer.stats(), prefix="timer")

            # replay metrics
            if isinstance(replay.sampler, PrioritizedSampler) and PRIORITY_KEY in mets:
                priorities_dist = [
                    replay.sampler.key2priority[key]
                    for key in uuid_keys
                    if replay.sampler.has_key(key)
                ]
                # if hasattr(train_replay, "sampler_pos"):
                if config.replay.imbalance == "upsample_pos":
                    priorities_dist += [
                        replay.sampler_pos.key2priority[key]
                        for key in uuid_keys
                        if replay.sampler_pos.has_key(key)
                    ]

                # len(priorities_dist) can be < len(uuid_keys) because some can be deleted before this
                # assert len(priorities_dist) == len(uuid_keys), (
                #     len(priorities_dist),
                #     len(uuid_keys),
                # )

                priorities_dist = np.array(priorities_dist).reshape(-1)
                # take from priorities_dist all elems < 50
                priorities_dist_loss = priorities_dist[priorities_dist < 50]

                priority_stats = {
                    "mean_priorities": np.mean(priorities_dist),
                    "mean_sample": recent_mean_seq_sample,
                    "priorities_dist": priorities_dist,
                    "priorities_dist_loss": priorities_dist_loss,
                }
            else:
                priority_stats = {}

            logger.add(
                {
                    **replay.stats,
                    **priority_stats,
                    "duplicated_keys": num_duplicated_keys,
                },
                prefix="replay",
            )

            duration = time.time() - stats["last_time"]
            actor_fps = (int(step) - stats["last_step"]) / duration
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
                last_step=int(step),
                batch_entries=0,
                last_real_step=real_env_step.value,
                last_opt_step=opt_step.value,
            )
            logger.write(fps=True)

        if not config.run.debug and should_save():
            checkpoint.save()


# from ledwm.embodied.core.distr import MAX_RECONNECT, MAX_RECONNECT_LOAD


def need_longer_time_to_load(args):
    return args.resume != "" or args.debug or args.script == "parallel_eval"


def print_eps_info(env_id, length, score, env_addr_types):
    env_address = get_env_address(env_id)
    env_type = env_addr_types[env_address]

    if env_type == "train":
        print(f"[{env_id}] Episode of length {length} with score {score:.4f}.")

    elif env_type == "eval":
        cprint(f"[{env_id}]", "blue", end="")
        print(f"Episode of length {length} with score {score:.4f}.")

    elif env_type == "test":
        cprint(f"[{env_id}]", "green", end="")
        print(f"Episode of length {length} with score {score:.4f}.")

    else:
        raise ValueError(f"Unknown env type: {env_type}")


def parallel_env(env_id, make_env, args, env_addr_types, timer=None):
    assert env_id >= 0, env_id
    print(f"[{env_id}] Make env.")
    env = make_env()
    timer and timer.wrap("env", env, ["step"])  # type: ignore
    addr = f"{args.actor_host}:{args.actor_port}"

    actor = distr.Client(
        addr,
        env_id,
        args.ipv6,
        timeout=distr.resolve_actor_timeout(args),
        # max_reconnect=(
        #     MAX_RECONNECT_LOAD if need_longer_time_to_load(args) else MAX_RECONNECT
        # ),
        max_reconnect=args.max_reconnect,
    )
    done = True
    act = None
    start = time.time()
    score = length = count = 0

    while True:
        if done:
            act = {k: v.sample() for k, v in env.act_space.items()}
            act["reset"] = True
            score, length = 0, 0

        start_timer = time.time()
        try:
            obs = env.step(act)
        except Exception as e:
            raise ValueError(f"Error in env.step: {e}")

        obs = {k: np.asarray(v) for k, v in obs.items()}
        score += obs["reward"]
        length += 1  # reset counts as a step in here. SO length == 1 means reset

        done = obs["is_last"]
        if done:
            print_eps_info(env_id, length, score, env_addr_types)
            if args.first_step:
                assert length == 1, length

        promise = actor(obs)
        try:
            act = promise()
            act = {k: v for k, v in act.items() if not k.startswith("log_")}

        except distr.ReconnectError:
            print(f"[{env_id}] Starting new episode because the client reconnected.")
            done = True

        except distr.RemoteError as e:
            print(f"[{env_id}] Shutting down env due to agent error: {e}")
            sys.exit(0)

        count += 1
        now = time.time()
        if now - start >= 60:
            fps = count / (now - start)
            print(f"[{env_id}] Env steps per second: {fps:.1f}")
            start = now
            count = 0


def dummy_data(spaces, batch_dims):
    # TODO: Get rid of this function by adding initial_policy_state() and
    # initial_train_state() to the agent API.
    spaces = list(spaces.items())
    data = {k: np.zeros(v.shape, v.dtype) for k, v in spaces}
    for dim in reversed(batch_dims):
        data = {k: np.repeat(v[None], dim, axis=0) for k, v in data.items()}
    return data
