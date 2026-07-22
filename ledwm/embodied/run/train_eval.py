from typing import TYPE_CHECKING
from termcolor import cprint

if TYPE_CHECKING:
    from ledwm.embodied.replay.generic import GenericReplay
from ledwm.agent import Agent
import numpy as np

from ledwm.embodied.core.checkpoint import Checkpoint
from ledwm.embodied.core.counter import Counter
from ledwm.embodied.core.driver import Driver
from ledwm.embodied.core.metrics import Metrics
from ledwm.embodied.core.path import Path
from ledwm.embodied.core.random import RandomAgent
from ledwm.embodied.core.timer import Timer
from ledwm.embodied.core.when import Clock, Every, Ratio, Until
from ledwm.embodied.core.basics import format as format_
import re


def train_eval(
    agent: "Agent",
    train_env,
    eval_env,
    train_replay: "GenericReplay",
    eval_replay: "GenericReplay",
    logger,
    args,
):
    logdir = Path(args.logdir)
    logdir.mkdirs()
    print("Logdir", logdir)
    should_expl = embodied.when.Until(args.expl_until)
    should_train = Ratio(args.train_ratio / args.batch_steps)
    should_log = Clock(args.log_every)
    should_save = Clock(args.save_every)
    should_eval = embodied.when.Every(args.eval_every, args.eval_initial)
    # should_sync = embodied.when.Every(args.sync_every)
    step = logger.step
    updates = Counter()
    metrics = Metrics()
    print("Observation space:", format_(train_env.obs_space), sep="\n")
    print("Action space:", format_(train_env.act_space), sep="\n")

    timer = embodied.Timer()
    timer.wrap("agent", agent, ["policy", "train", "report", "save"])
    timer.wrap("env", train_env, ["step"])
    if hasattr(train_replay, "_sample"):
        timer.wrap("replay", train_replay, ["_sample"])

    nonzeros = set()
    wins = []

    def per_episode(ep, env_id, mode):
        length = len(ep["reward"]) - 1
        score = float(ep["reward"].astype(np.float64).sum())
        logger.add(
            {
                "length": length,
                "score": score,
                "reward_rate": (ep["reward"] - ep["reward"].min() >= 0.1).mean(),
            },
            prefix=("episode" if mode == "train" else f"{mode}_episode"),
        )

        if mode == "eval":
            wins.append(score > 0)

        print(f"[{env_id}][{mode}] Episode has {length} steps and return {score:.1f}.")
        stats = {}
        for key in args.log_keys_video:
            if key in ep:
                stats[f"policy_{key}"] = ep[key]

        for key, value in ep.items():
            if not args.log_zeros and key not in nonzeros and (value == 0).all():
                continue
            nonzeros.add(key)
            if re.match(args.log_keys_sum, key):
                stats[f"sum_{key}"] = ep[key].sum()
            if re.match(args.log_keys_mean, key):
                stats[f"mean_{key}"] = ep[key].mean()
            if re.match(args.log_keys_max, key):
                stats[f"max_{key}"] = ep[key].max(0).mean()

        metrics.add(stats, prefix=f"{mode}_stats")

    cprint(f"Len train env: {len(train_env)}", "green")
    driver_train = Driver(train_env)
    driver_train.on_episode(lambda ep, env_id: per_episode(ep, env_id, mode="train"))
    driver_train.on_step(lambda tran, _: step.increment())
    driver_train.on_step(train_replay.add)
    driver_eval = Driver(eval_env)
    driver_eval.on_step(eval_replay.add)
    driver_eval.on_episode(lambda ep, env_id: per_episode(ep, env_id, mode="eval"))

    random_agent = RandomAgent(train_env.act_space)
    print(
        f"Prefill train dataset until {max(args.batch_steps, args.train_fill)} steps."
    )
    fill_steps = max(args.batch_steps, args.train_fill)
    while len(train_replay) < fill_steps:
        # if len(train_replay) % 1000 == 0:
        #     print(f"len(train_replay): {len(train_replay)} / {fill_steps}")
        driver_train.__call__(random_agent.policy, max_steps=100)

    fill_steps = max(args.batch_steps, args.eval_fill)
    print(f"Prefill eval dataset until {fill_steps} steps.")
    while len(eval_replay) < fill_steps:
        # if len(eval_replay) % 1000 == 0:
        #     print(f"len(eval_replay): {len(eval_replay)} / {fill_steps}")
        driver_eval.__call__(random_agent.policy, max_steps=100, mode="eval")

    logger.add(metrics.result())
    logger.write()

    dataset_train = agent.dataset(train_replay.dataset)
    dataset_eval = agent.dataset(eval_replay.dataset)
    state = [None]  # To be writable from train step function below.
    batch = [None]

    def train_step(tran, worker):
        for _ in range(should_train(step)):
            with timer.scope("dataset_train"):
                batch[0] = next(dataset_train)

            opt_step.increment()
            outs, state[0], mets = agent.train(batch[0], state[0], step=opt_step)
            metrics.add(mets, prefix="train")

            if "priority" in outs:
                train_replay.prioritize(outs["key"], outs["priority"])
            updates.increment()

        # if should_sync(updates):
        #     agent.sync() # TODO check again in dreamerv3 - https://github.com/danijar/dreamerv3/blob/8fa35f83eee1ce7e10f3dee0b766587d0a713a60/dreamerv3/jaxagent.py#L118

        if should_log(step):
            logger.add(metrics.result())
            logger.add(agent.report(batch[0], step=opt_step), prefix="report")
            with timer.scope("dataset_eval"):
                eval_batch = next(dataset_eval)
            logger.add(agent.report(eval_batch, step=opt_step), prefix="eval")
            logger.add(train_replay.stats, prefix="replay")
            logger.add(eval_replay.stats, prefix="eval_replay")
            logger.add(timer.stats(), prefix="timer")
            logger.write(fps=True)

    driver_train.on_step(train_step)

    opt_step = Counter()
    checkpoint = Checkpoint(logdir / "checkpoint.ckpt")
    checkpoint.step = step
    checkpoint.agent = agent
    checkpoint.train_replay = train_replay
    checkpoint.eval_replay = eval_replay
    checkpoint.opt_step = opt_step

    if args.from_checkpoint:
        checkpoint.load(args.from_checkpoint)

    checkpoint.load_or_save()
    should_save(step)  # Register that we jused saved.

    print("Start training loop.")
    policy_train = lambda *args: agent.policy(
        *args, mode="explore" if should_expl(step) else "train", step=opt_step
    )
    policy_eval = lambda *args: agent.policy(*args, mode="eval", step=opt_step)

    while step < args.steps:
        if should_eval(step):
            cprint(f"Starting evaluation at step {int(step)}", "green")
            driver_eval.reset()
            driver_eval.__call__(
                policy_eval,
                max_episodes=max(len(eval_env), args.eval_eps),
                mode="eval",
                opt_step=opt_step,
            )
            win_rate = np.mean(wins)
            cprint(
                f"Win rates (win_rates) over {len(wins)} episodes = {win_rate}", "green"
            )
            logger.add({"win_rate": win_rate}, prefix="eval")
            # clear wins
            wins = []

        driver_train.__call__(policy_train, max_steps=100, opt_step=opt_step)
        if should_save(step):
            checkpoint.save()

    cprint(f"Done at {step=}", "green")
    logger.write()
    logger.write()
