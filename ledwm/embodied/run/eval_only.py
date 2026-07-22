import re
from typing import TYPE_CHECKING, List


import collections

from termcolor import cprint

if TYPE_CHECKING:
    from ledwm.agent import Agent
    from ledwm.embodied.core.logger import Logger
    from ledwm.embodied.core.counter import Counter
    from ledwm.embodied.replay.replays import Uniform
import numpy as np

from ledwm.embodied.core.checkpoint import Checkpoint
from ledwm.embodied.core.counter import Counter
from ledwm.embodied.core.driver import Driver
from ledwm.embodied.core.metrics import Metrics
from ledwm.embodied.core.path import Path
from ledwm.embodied.core.timer import Timer
from ledwm.embodied.core.when import Clock


def save_frames(ep, outdir, img_key="image", text_key="token"):
    from PIL import Image
    import os

    os.makedirs(outdir, exist_ok=True)
    #  print(f"Saving ep with seq len {len(ep[img_key])}")
    for t, frame in enumerate(ep[img_key]):
        im = Image.fromarray(frame)
        im.save(f"{outdir}/{t}.png")
    with open(f"{outdir}/tokens.txt", "w") as f:
        for tok in ep[text_key]:
            f.write(f"{tok}\n")
    with open(f"{outdir}/rewards.txt", "w") as f:
        for tok in ep["reward"]:
            f.write(f"{tok}\n")
    with open(f"{outdir}/actions.txt", "w") as f:
        for tok in ep["action"]:
            f.write(f"{tok}\n")
    if "log_language_info" in ep:
        with open(f"{outdir}/lang.txt", "w") as f:
            for tok in ep["log_language_info"]:
                f.write(f"{tok}\n")


def done_cal_win_rate(win_rates: List[float], score: float, episode: "Counter", config):
    # WIN_REWARD = 1.5
    # MAX_NUM_EPISODES = 1000
    WIN_REWARDS = {"s1": 1, "s2": 1.5, "s3": 1.5}
    win_rates.append(score)
    episode.increment()
    WIN_REWARDS = {"s1": 1, "s2": 1.5, "s3": 1.5}

    if episode == config.run.eval_eps:
        assert len(win_rates) == config.run.eval_eps
        win_rate = WIN_REWARDS[config.task.split("_")[1]]
        cprint(f"FINAL win_rates= {np.mean(np.array(win_rates) == win_rate)}", "green")
        print("Reward Counter:", collections.Counter(win_rates))
        return True

    return False


def eval_inference_only(
    agent: "Agent",
    env,
    replay: "Uniform",
    logger: "Logger",
    args,
):
    logdir = Path(args.logdir)
    logdir.mkdirs()
    print("Logdir", logdir)

    should_log = Clock(args.log_every)
    step = logger.step
    eps_step = Counter()
    metrics = Metrics()
    print("Observation space:", env.obs_space)
    print("Action space:", env.act_space)
    WIN_REWARDS = {"s1": 1, "s2": 1.5, "s3": 1.5}
    scores = []

    def per_episode(ep):
        length = len(ep["reward"]) - 1
        score = float(ep["reward"].astype(np.float64).sum())
        logger.add({"length": length, "score": score}, prefix="episode")
        print(f"Episode has {length} steps and return {score:.1f}.")
        scores.append(score)

        # stats = {}
        # for key in args.log_keys_video:
        #     if key in ep:
        #         stats[f"policy_{key}"] = ep[key]

        # for key, value in ep.items():
        #     if not args.log_zeros and key not in nonzeros and (value == 0).all():
        #         continue
        #     nonzeros.add(key)
        #     if re.match(args.log_keys_sum, key):
        #         stats[f"sum_{key}"] = ep[key].sum()
        #     if re.match(args.log_keys_mean, key):
        #         stats[f"mean_{key}"] = ep[key].mean()
        #     if re.match(args.log_keys_max, key):
        #         stats[f"max_{key}"] = ep[key].max(0).mean()
        # metrics.add(stats, prefix="stats")

    def inference(trans, worker):
        for _ in range(should_train(step)):
            with timer.scope("dataset_train"):
                batch[0] = next(dataset_train)
            outs, state[0], mets = agent.train(batch[0], state[0])
            metrics.add(mets, prefix="train")

    driver = Driver(env)
    driver.on_episode(lambda ep, worker: per_episode(ep))
    driver.on_episode(lambda *args: eps_step.increment())
    driver.on_step(lambda tran, _: step.increment())
    driver.on_step(replay.add)
    driver.on_step(inference)

    checkpoint = Checkpoint()
    checkpoint.agent = agent
    checkpoint.load(args.from_checkpoint, keys=["agent"])

    print("Start evaluation loop.")
    policy = lambda *args: agent.policy(*args, mode="eval")
    dataset = agent.dataset(replay.dataset)
    driver.__call__(policy, max_episodes=args.num_val_episodes)


def eval_only(agent: "Agent", env, logger: "Logger", args):
    logdir = Path(args.logdir)
    logdir.mkdirs()
    print("Logdir", logdir)
    should_log = Clock(args.log_every)
    should_print_win_rate = Clock(10)
    step = logger.step
    metrics = Metrics()
    print("Observation space:", env.obs_space)
    print("Action space:", env.act_space)

    timer = Timer()
    timer.wrap("agent", agent, ["policy"])
    timer.wrap("env", env, ["step"])
    timer.wrap("logger", logger, ["write"])

    nonzeros = set()
    fails = 0
    succs = 0
    thres = 3
    episode = Counter()
    win_rates = []

    def per_episode(ep):
        length = len(ep["reward"]) - 1
        score = float(ep["reward"].astype(np.float64).sum())
        logger.add({"length": length, "score": score}, prefix="episode")
        print(f"Episode has {length} steps and return {score:.1f}.")
        # if done_cal_win_rate(win_rates, score, episode):
        # exit()

        nonlocal fails
        nonlocal succs

        if score <= thres:
            if fails < 3:
                path = f"{args.save_frames_to}/fail{fails}"
                fails += 1
                save_frames(ep, f"{path}", img_key="log_image")
        else:
            path = f"{args.save_frames_to}/succ{succs}"
            succs += 1
            save_frames(ep, f"{path}", img_key="log_image")

        if succs >= 4:
            exit()

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
        metrics.add(stats, prefix="stats")

    driver = Driver(env)
    driver.on_episode(lambda ep, worker: per_episode(ep))
    driver.on_step(lambda tran, _: step.increment())

    checkpoint = Checkpoint()
    checkpoint.agent = agent
    checkpoint.load(args.from_checkpoint, keys=["agent"])

    print("Start evaluation loop.")
    policy = lambda *args: agent.policy(*args, mode="eval")

    while step < args.steps:
        driver.__call__(policy, max_steps=100)
        if should_log.__call__(step):
            logger.add(metrics.result())
            logger.add(timer.stats(), prefix="timer")
            logger.write(fps=True)

    logger.write()
