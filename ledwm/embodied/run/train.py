# import collections
import re
from termcolor import cprint
from tqdm import tqdm
from ledwm.embodied.core.WandBOutput import make_table_data_text
from ledwm.embodied.core.driver import Driver
from ledwm.embodied.core.OracleAgent import OracleAgent
from ledwm.embodied.core.random import RandomAgent
import numpy as np

from ledwm.embodied.core.checkpoint import Checkpoint
from ledwm.embodied.core.counter import Counter
from ledwm.embodied.core.metrics import Metrics
from ledwm.embodied.core.path import Path
from ledwm.embodied.core.timer import Timer
from ledwm.embodied.core.usage import Usage
from ledwm.embodied.core.when import Clock, Ratio, Until
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from ledwm.embodied.replay.curious_replay import CuriousReplay
    from ledwm.agent import Agent
    from ledwm.embodied.replay.generic import GenericReplay
    from ledwm.embodied.replay.replays import Uniform
    from ledwm.embodied.run.smoothing import ReplayEps
    from ledwm.embodied.core.logger import Logger


def train(
    agent: "Agent | RandomAgent | OracleAgent",
    env,
    replay: "GenericReplay | Uniform | ReplayEps | CuriousReplay",
    logger: "Logger",
    args,
):
    logdir = Path(args.logdir)
    logdir.mkdirs()
    print("Logdir", logdir)
    should_expl = Until(args.expl_until)
    should_train = Ratio(args.train_ratio / args.batch_steps)
    should_log = Clock(args.log_every)
    should_save = Clock(args.save_every)
    usage = Usage(args.trace_malloc)
    step = logger.step

    # Env steps without reading
    real_env_step = Counter()
    metrics = Metrics()
    print("Observation space:")
    for key, value in env.obs_space.items():
        print(f"  {key:<16} {value}")
    print("Action space:")
    for key, value in env.act_space.items():
        print(f"  {key:<16} {value}")

    timer = Timer()
    timer.wrap("agent", agent, ["policy", "train", "report", "save"])
    timer.wrap("env", env, ["step"])
    timer.wrap("replay", replay, ["add", "save"])
    timer.wrap("logger", logger, ["write"])
    nonzeros = set()

    def per_episode(ep):
        length = len(ep["reward"]) - 1
        score = float(ep["reward"].astype(np.float64).sum())
        sum_abs_reward = float(np.abs(ep["reward"]).astype(np.float64).sum())
        logger.add(
            {
                # "real_length": len(ep["is_read_step"]) - sum(ep["is_read_step"]),
                "length": length,
                "score": score,
                "sum_abs_reward": sum_abs_reward,
                "reward_rate": (np.abs(ep["reward"]) >= 0.5).mean(),
            },
            prefix="episode",
        )
        logger.add({"real_step": real_env_step.value})
        print(f"Episode has {length} steps and return {score:.1f}.")
        if args.first_step:
            assert length == 0, length
        stats = {}

        if args.log_video:
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

    def count_real_step(tran):
        if not tran.get("is_read_step", False):
            real_env_step.increment()

    # TODO check it later data_exclude_keys
    driver = Driver(env, first_step=args.first_step, exclude_keys=[])
    driver.on_episode(lambda ep, worker: per_episode(ep))
    driver.on_step(lambda tran, _: step.increment())
    driver.on_step(lambda tran, _: count_real_step(tran))
    driver.on_step(replay.add)

    random_agent = RandomAgent(env.act_space)
    if args.overfit_eps:
        fill_steps = args.train_fill - len(replay)
    else:
        fill_steps = max(args.batch_steps, args.train_fill - len(replay))

    print(f"Fill train dataset ({fill_steps} steps).")
    if args.overfit_eps:
        assert isinstance(replay, Uniform)
        assert replay.size >= fill_steps, (
            f"capacity {replay.size} < fill_steps {fill_steps}"
        )

    while len(replay) < fill_steps:
        driver.__call__(random_agent.policy, max_steps=args.train_steps)
    logger.add(metrics.result())
    logger.write()

    dataset = agent.dataset(replay.dataset)
    state = [None]  # To be writable from train step function below.
    assert args.pretrain > 0  # At least one step to initialize variables.

    checkpoint = Checkpoint(logdir / "checkpoint.ckpt", not args.overfit_eps)
    timer.wrap("checkpoint", checkpoint, ["save", "load"])
    checkpoint.step = step
    checkpoint.real_step = real_env_step
    checkpoint.agent = agent
    # checkpoint.replay = replay
    opt_step = Counter()

    if args.from_checkpoint != "":
        checkpoint.load(args.from_checkpoint)

    # if args.load_checkpoint:
    # checkpoint.load_or_save()
    should_save(step)  # Register that we jused saved.

    if args.overfit_eps:
        with timer.scope("dataset"):
            batch = next(dataset)

        for pretrain_iter in tqdm(range(args.pretrain)):
            _, state[0], mets = agent.train(batch, state[0])
            metrics.add(mets, prefix="train")

            # if should_log(step):
            if pretrain_iter % 10 == 0:
                agg = metrics.result()
                if "train/image_loss_mean" in agg:
                    if agg["train/image_loss_mean"] < 0.01:
                        cprint(f"Early stopping at iter {pretrain_iter}.", "green")
                        break

                logger.add({"real_step": real_env_step.value})
                logger.add(agg)
                logger.add(replay.stats, prefix="replay")
                if replay.stats["insert_wait_frac"] > 0.2:
                    cprint(
                        f"Insert wait frac: {replay.stats['insert_wait_frac']}", "red"
                    )
                if replay.stats["sample_wait_frac"] > 0.2:
                    cprint(
                        f"Sample wait frac: {replay.stats['sample_wait_frac']}", "red"
                    )

                logger.add(timer.stats(), prefix="timer")
                logger.add(usage.stats(), prefix="usage")
                logger.write(fps=True)
                checkpoint.save()

        cprint(f"Overfitting on {args.overfit_eps} episodes.", "green")
        checkpoint.save()
        return
    else:
        # not overfit
        assert len(replay) > 0, "Replay buffer is empty."
        with timer.scope("dataset"):
            batch = next(dataset)
        opt_step.increment()  # optimize_step
        _, state[0], mets = agent.train(batch, state[0], opt_step.value)

    batch = [None]

    def train_step(tran, worker):
        assert len(replay) > 0, "Replay buffer is empty."
        for _ in range(should_train(step)):
            with timer.scope("dataset"):
                batch[0] = next(dataset)

            opt_step.increment()  # optimize_step
            outs, state[0], mets = agent.train(batch[0], state[0], opt_step.value)
            metrics.add(mets, prefix="train")

            if "priority" in outs:
                assert not isinstance(replay, CuriousReplay)
                replay.prioritize(outs["key"], outs["priority"])

        if should_log(step):
            agg = metrics.result()
            report = agent.report(batch[0], opt_step.value)
            report = {
                k: v for k, v in report.items() if "train/" + k not in agg
            }  # Remove duplicates
            logger.add(agg)
            if args.use_table:
                report.update(make_table_data_text(report))
            logger.add(report, prefix="report")
            logger.add(replay.stats, prefix="replay")
            logger.add(timer.stats(), prefix="timer")
            logger.add(usage.stats(), prefix="usage")
            logger.add({"real_step": real_env_step.value})
            logger.write(fps=True)

    driver.on_step(train_step)

    print("Start training loop.")

    def policy(*args_fn):
        return agent.act(
            *args_fn,
            step=opt_step.value if args.opt_step else None,
            mode="explore" if should_expl(step) else "train",
        )

    # else:
    #     policy = random_agent.policy

    while step < args.steps:
        driver.__call__(policy, max_steps=args.train_steps)
        if should_save(step):
            checkpoint.save()
