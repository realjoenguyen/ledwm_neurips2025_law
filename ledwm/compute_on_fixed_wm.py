# %%
"""
before: before finetune
after: after finetune
"""

import collections
from functools import partial
import pathlib
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from termcolor import cprint
from typing import TYPE_CHECKING, List, Optional

import embodied
import pandas as pd
from ledwm.common import Timing
from embodied.core.basics import convert
from embodied.core.checkpoint import Checkpoint
from embodied.core.driver import Driver
from embodied.replay.replays import make_replay
from embodied.run.parallel_eval import WIN_REWARDS
from matplotlib.markers import MarkerStyle
from tqdm import tqdm

import wandb
from ledwm.embodied.core.logger import Logger
from ledwm.embodied.core.timer import Timer
from embodied.run.smoothing import ReplayEps

if TYPE_CHECKING:
    from ledwm.agent import Agent

IGNORE_KEYS = ["step_time", "id"]


def avg_discounted_return(
    reward,  # horizon+1, bs,
    cont,  # horizon+1, bs
    discount=1,
    # debug=False,
):
    # Compute discount factors for the entire sequence
    assert reward.shape == cont.shape, (reward.shape, cont.shape)
    timesteps = reward.shape[0]
    discount_factors = discount ** np.arange(timesteps)
    # print(f"{discount_factors=}")
    in_eps = compute_in_eps(cont)
    pred_rewards = reward[in_eps > 0]

    # print(pred_rewards)
    plot_hist_rewards_pred(pred_rewards)
    plot_rewards_pred_freq(reward, in_eps)

    pred_reward_sums = (reward * in_eps * discount_factors[:, np.newaxis]).sum(axis=0)
    return pred_reward_sums.mean()


def test_compute_discounted_reward_sum():
    # Define test parameters
    horizon = 5
    bs = 5  # batch size

    # Generate dummy rewards in the range [0, -1, 0.5, 1]
    np.random.seed(42)  # Ensure reproducibility
    reward = np.random.choice([0, -1, 0.5, 1], size=(horizon + 1, bs))

    # Generate dummy continuation mask (0 means end)
    cont = np.random.choice([1, 0], size=(horizon + 1, bs), p=[0.8, 0.2])

    # # Ensure at least one `1` in each episode to simulate an end
    # cont[np.random.randint(0, horizon), np.arange(bs)] = (
    #     1  # Ensure an end in each episode
    # )

    # Run function
    result = avg_discounted_return(reward, cont)

    # Print test case information
    print("Reward Matrix:\n", reward)
    print("Continuation Matrix:\n", cont)
    print("Computed Discounted Reward Sum:", result)

    # Simple assertions
    assert isinstance(result, float), "Output should be a float"
    assert not np.isnan(result), "Output should not be NaN"
    assert result <= np.max(reward.sum(axis=0)), "Should not exceed max possible return"
    assert result >= np.min(reward.sum(axis=0)), (
        "Should not be less than min possible return"
    )


def p_test(before_means: List[float], after_means: List[float]):
    diffs = [b - a for b, a in zip(before_means, after_means)]
    filtered_pairs = [
        (b, a) for b, a, d in zip(before_means, after_means, diffs) if d != 0
    ]

    filtered_before, filtered_after = (
        zip(*filtered_pairs) if filtered_pairs else ([], [])
    )
    if len(filtered_before) > 0:
        assert len(filtered_before) == len(filtered_after), (
            len(filtered_before),
            len(filtered_after),
        )
        _, p_value = stats.wilcoxon(filtered_before, filtered_after)
        THRESHOLD_P_VALUE = 0.05
        if p_value < THRESHOLD_P_VALUE:
            cprint("The difference is statistically significant.", "green")
        else:
            cprint(
                f"{p_value=} The difference is not statistically significant.",
                "red",
            )


def bootstrap(before_means: List[float], after_means: List[float]):
    bootstrap_means = []
    # diffs = after_means - before_means
    diffs = [a - b for a, b in zip(after_means, before_means)]
    BIG_ENOUGH_SAMPLES = 10000
    for i in range(BIG_ENOUGH_SAMPLES):
        bootstrap_means.append(np.random.choice(diffs, len(diffs), replace=True).mean())

    cprint(
        f"95% CI {np.percentile(bootstrap_means, 2.5)}, {np.percentile(bootstrap_means, 97.5)}",
        "green",
    )


def hier_bootstrap(
    before_eps_scores: List[List[float]], after_eps_scores: List[List[float]]
):
    BIG_ENOUGH_SAMPLES = 10000
    diffs = [
        [a - b for a, b in zip(after, before)]
        for before, after in zip(before_eps_scores, after_eps_scores)
    ]
    bstr_means = []

    for i in range(BIG_ENOUGH_SAMPLES):
        # sampling each game
        bstr_eps_means = []
        for eps_diff in diffs:
            sample_diffs = np.random.choice(eps_diff, len(eps_diff), replace=True)
            bstr_eps_means.append(sample_diffs.mean())

        bstr_means.append(
            np.random.choice(bstr_eps_means, len(bstr_eps_means), replace=True).mean()
        )

    ci_lower, ci_upper = np.percentile(bstr_means, [2.5, 97.5])
    cprint(f"Hierarchical 95% CI {ci_lower}, {ci_upper}", "green")


def test_policy(agent: "Agent", *args_fn):
    return agent.policy(*args_fn, mode="test")


def measure_return(
    agent: "Agent",
    eval_envs,
    logger,
    checkpoint: "Checkpoint",
    config,
    skip_successful=False,
):
    print(f"Measure return. {skip_successful=}")
    args = config.run
    real_return_means = []
    imag_returns = []

    WIN_RATES = [0, 0.25, 0.5, 0.75, 1]
    # WIN_RATES = [0.25, 0.5, 0.75, 1]
    # WIN_RATES = [1]
    cp_dir = pathlib.Path(args.from_checkpoint).parent

    NUM_ENV_TEST = len(eval_envs)
    cprint(f"Evaluate on {NUM_ENV_TEST=} envs", "green")
    driver_real = Driver(eval_envs)

    real_scores_per_game = []
    worker2scores = collections.defaultdict(list)
    worker2actions = collections.defaultdict(list)
    driver_real.on_episode(
        lambda ep, env_id: per_episode(
            ep,
            env_id,
            real_scores_per_game,
            logger=logger,
            args=args,
            mode="eval",
        )
    )
    driver_real.on_episode(
        lambda ep, env_id: worker2scores[env_id].append(ep["reward"].sum())
    )
    driver_real.on_episode(
        lambda ep, env_id: worker2actions[env_id].append(ep["action"])
    )

    replay = collections.defaultdict(list)

    # replay = None
    def add_to_replay(replay, step_data, worker, *args):
        if replay is not None:
            replay[worker].append(step_data)

    driver_real.on_step(partial(add_to_replay, replay))

    for win_rate in WIN_RATES:
        path = cp_dir / f"checkpoint_{win_rate}.ckpt"
        if win_rate == 1 and not path.exists():
            path = cp_dir / "checkpoint.ckpt"
        if not path.exists():
            cprint(f"{path} does not exist. Skip this checkpoint.", "red")

        eps_each_win_rate = args.eps_finetune // len(WIN_RATES)
        num_eps = 0

        for win_rate in WIN_RATES:
            path = cp_dir / f"checkpoint_{win_rate}.ckpt"
            checkpoint.load(
                path, configs={"agent": {"load_only_key": "agent/task_behavior"}}
            )

            for sample_iter in tqdm(
                range(eps_each_win_rate),
                desc=f"Measure return when policy win_rate = {win_rate}",
            ):
                num_eps += 1
                game_config = eval_envs.create_game_config()
                eval_envs.reset_game_config(**game_config)

                # with Timing("Cal before scores", config.run.debug):
                # clear replay
                for k in replay:
                    replay[k].clear()
                for k in worker2scores:
                    worker2scores[k].clear()
                for k in worker2actions:
                    worker2actions[k].clear()

                real_scores_per_game.clear()
                driver_real.reset()
                driver_real.__call__(
                    partial(test_policy, agent),
                    max_episodes=NUM_ENV_TEST,
                    opt_step=sample_iter if config.run.opt_step else None,
                    config=config,
                )
                assert len(real_scores_per_game) == NUM_ENV_TEST, (
                    len(real_scores_per_game),
                    NUM_ENV_TEST,
                )
                # test

                if "s1" in config.task:
                    assert all(e in [-1, 1] for e in real_scores_per_game), (
                        real_scores_per_game
                    )
                    assert all(
                        e == real_scores_per_game[0] for e in real_scores_per_game
                    ), set(real_scores_per_game)

                real_return_mean = np.mean(real_scores_per_game)
                real_return_means.append(real_return_mean)
                print("Mean of real scores =", np.mean(real_scores_per_game))

                first_obsss = compute_first_obsss(eval_envs)
                batch = agent.postprocess(first_obsss)  # shape: (bs, bl, d*)
                pred_return = avg_imag_return(agent, batch)
                imag_returns.append(pred_return)
                return_mean = np.mean(real_scores_per_game)

                print(
                    f"real: {return_mean}, pred: {pred_return}",
                )

                if sample_iter % 100 == 0 and sample_iter > 0:
                    if "lwm" in config.task:
                        mode = config.env.lwm.mode
                    else:
                        mode = config.env.messenger.mode

                    assert max(real_return_means) <= WIN_REWARDS[config.task], max(
                        real_return_means
                    )
                    assert min(real_return_means) >= -1, min(real_return_means)

                    plot_value_gap(
                        real_return_means,
                        imag_returns,
                        checkpoint=pathlib.Path(config.run.from_checkpoint).parent.name,
                        num_eps=num_eps,
                        config=config,
                        mode=config.measure_policy + "_" + config.task + "_" + mode,
                    )
                    # plot scatter before_means_real_vs_imag
                    plot_scatter_before_means_real_vs_imag(
                        real_return_means,
                        imag_returns,
                    )


def plot_value_gap(
    real_means: List[float],
    imag_means: List[float],
    checkpoint,
    num_eps: int,
    config,
    step_finetune=0,
    skip_win=False,
    skip_from_wm=False,
    lose_only=False,
    condition_wm="",
    mode="",
):
    assert len(real_means) == len(imag_means), (
        len(real_means),
        len(imag_means),
    )
    val_gap_abs = abs(np.array(imag_means) - np.array(real_means))
    data = pd.DataFrame({"before_means": real_means, "value_gap_abs": val_gap_abs})
    bins_length_0_2 = np.arange(-1, WIN_REWARDS[config.task], 0.2)
    bins_length_0_2[-1] = WIN_REWARDS[config.task] + 0.1
    bins_length_0_2[0] = -1.1
    # print(f"{bins_length_0_2=}")

    labels_right_bound = [
        f"{round(bins_length_0_2[i + 1], 2)}" for i in range(len(bins_length_0_2) - 1)
    ]
    # print(labels_right_bound)
    data["bins_length_0_2"] = pd.cut(
        data["before_means"],
        bins=list(bins_length_0_2),
        labels=labels_right_bound,
        right=True,
        # data["before_means"], bins=15, labels=labels_right_bound
    )

    # Recalculate mean differences within each 0.2-length bin
    mean_differences_length_0_2 = data.groupby("bins_length_0_2")[
        "value_gap_abs"
    ].mean()

    # Calculate frequencies of before_means in each bin
    frequency_before_means = data["bins_length_0_2"].value_counts().sort_index()
    colors_length_0_2 = [
        "green" if diff > 0 else "red" for diff in mean_differences_length_0_2
    ]
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Bar plot for mean differences, shifted to the left within each bin
    x_positions = np.arange(len(mean_differences_length_0_2))
    ax1.bar(
        x_positions - 0.2,
        mean_differences_length_0_2,
        color=colors_length_0_2,
        width=0.4,
        label="Mean Difference",
    )
    # caption
    # avg of differences
    val_gap_avg = np.mean(val_gap_abs)
    ax1.set_xlabel(f"{num_eps=}, {val_gap_avg=}")
    ax1.set_ylabel("Mean Difference (After - Before)")
    ax1.set_xticks(x_positions)
    ax1.set_xticklabels(mean_differences_length_0_2.index, rotation=0)
    ax1.set_title(f"Value Gap. {mode=}, {num_eps=}")
    ax1.set_ylim(
        bottom=min(0, mean_differences_length_0_2.min() * 1.5),
        top=max(0.2, mean_differences_length_0_2.max() * 1.5),
    )

    # Secondary Y-axis for frequency on the right side
    ax2 = ax1.twinx()

    # find min x in [100, 200, 300, ..., 10000] s.t val / x < ax1.get_ylim()[1] -> x > val / ax1.get_ylim()[1] in [100, 200, 300, ..., 10000]
    possible_scales = np.arange(0, 20000, 10)
    found_scale = 0.01
    for scale in possible_scales:
        if scale > 0 and frequency_before_means.max() / scale < ax1.get_ylim()[1]:
            found_scale = scale
            print("scale frequencies by ", found_scale)
            break

    frequency_scaled = frequency_before_means / found_scale

    ax1.set_ylim(
        bottom=min(0, mean_differences_length_0_2.min() * 1.5),
        top=max(0.2, mean_differences_length_0_2.max() * 2),
    )
    ax2.bar(
        x_positions + 0.2,
        frequency_scaled,
        color="blue",
        alpha=0.3,
        width=0.4,
        label="Frequency",
    )
    ax2.set_ylim(
        ax1.get_ylim()  # type: ignore
    )  # Ensures the frequency axis has the same Y=0 line as the mean difference axis
    ax2.set_ylabel(f"Frequency of before_means. (*{found_scale})")

    # Add legend for clarity
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")
    file_path = (
        pathlib.Path("./results")
        / checkpoint
        / f"{mode}_{config.imag_horizon}_value_gap_{step_finetune}.png"
    )
    if not file_path.parent.exists():
        file_path.parent.mkdir(parents=True)
        print(f"Create directory {file_path.parent}")
    # plt.savefig(file_path)
    # print("Save plot to", file_path)
    fig = plt.gcf()
    plt.show()
    if wandb.run is not None:
        wandb.log({"value_gap": wandb.Image(fig)})
    plt.close()


import matplotlib.colors as mcolors


def plot_scatter_after_imag_real_diff_ft(
    imag_diffs: List[float],
    real_diffs: List[float],
    # before_real_returns. Indicates how much win for this score. From 0 to 1
    win_levels: List[float],
    checkpoint,
    num_eps: int,
    step_finetune: int,
    skip_win=False,
    skip_from_wm=False,
    lose_only=False,
    condition_wm="",
):
    # scatter plot of after_imag_diffs and after_real_diffs
    assert len(imag_diffs) == len(real_diffs) == len(win_levels), (
        len(imag_diffs),
        len(real_diffs),
        len(win_levels),
    )
    fig, ax = plt.subplots(figsize=(10, 6))

    # Create a colormap from green to red
    from matplotlib import cm

    cmap = cm.get_cmap("RdYlGn")
    norm = mcolors.Normalize(vmin=0, vmax=1)

    # Scatter plot with colors based on win_levels
    sc = ax.scatter(
        imag_diffs,
        real_diffs,
        c=win_levels,
        cmap=cmap,
        norm=norm,
        alpha=0.3,
        edgecolor="k",
        linewidth=0.5,
    )

    # Add colorbar
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("Win Levels")

    ax.axhline(0, color="black", lw=2)
    ax.axvline(0, color="black", lw=2)
    ax.set_xlabel("After Imag Diff")
    ax.set_ylabel("After Real Diff")
    ax.set_title(
        f"Scatter plot of After Imag Diff and After Real Diff. {skip_from_wm=}, {skip_win=}, {lose_only=}, {num_eps=}. \n{condition_wm=}. {step_finetune=}"
    )
    file_path = (
        pathlib.Path("./results/")
        / checkpoint
        / f"scatter_after_imag_real_diff_{lose_only=}, {skip_win=}_{skip_from_wm=}_{condition_wm=}_{step_finetune}.png"
    )
    if not file_path.parent.exists():
        file_path.parent.mkdir(parents=True)
        print(f"Create directory {file_path.parent}")
    # plt.savefig(file_path)
    # print("Save plot to", file_path)

    fig = plt.gcf()
    if wandb.run is not None:
        wandb.log({"scatter_after_imag_real_diff": wandb.Image(fig)})
    plt.close()


def plot_scatter_before_means_real_vs_imag(
    before_means: List[float], before_means_imag: List[float]
):
    assert len(before_means) == len(before_means_imag), (
        len(before_means),
        len(before_means_imag),
    )
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(
        before_means, before_means_imag, marker=MarkerStyle("o"), color="red", alpha=0.2
    )
    # show grid
    ax.grid(True)
    # draw line x=y
    ax.axhline(0, color="black", lw=2)
    ax.axvline(0, color="black", lw=2)
    ax.set_xlabel("Before Means")
    ax.set_ylabel("Before Means Imag")
    ax.set_title("Scatter plot of Before Means and Before Means Imag")
    # plt.show()
    fig = plt.gcf()
    if wandb.run is not None:
        wandb.log({"scatter_before_means_real_vs_imag": wandb.Image(fig)})
    plt.close()


def plot_histogram(wins, loses, metric, accum, checkpoint, num_eps):
    # Create the plot
    plt.figure(figsize=(10, 6))

    # Determine the bin edges for finer granularity
    min_value = min(wins) if not loses else min(min(wins), min(loses))
    max_value = max(wins) if not loses else max(max(wins), max(loses))
    bins = [round(x, 2) for x in np.arange(min_value, max_value, 0.1)]

    # Plot density for wins (normalized histogram)
    plt.hist(wins, bins=bins, density=True, alpha=0.6, color="green", label="Wins")

    # Plot density for losses (normalized histogram) only if 'loses' is not empty
    if loses:
        plt.hist(loses, bins=bins, density=True, alpha=0.6, color="red", label="Losses")

    # Add labels and title
    plt.xlabel("Value")
    plt.ylabel("Density")
    plt.title(
        f"Density of Win and Lose Values. {checkpoint=}, {len(wins)=}, {len(loses)=}, {metric=}"
    )

    # Add a legend to distinguish between wins and losses
    plt.legend()

    # Save the plot
    name = f"[{accum}] win_lose_{metric}_{checkpoint}.png"
    path = pathlib.Path(checkpoint) / name
    # If the path does not exist, create the checkpoint directory
    if not path.parent.exists():
        path.parent.mkdir(parents=True)
        print(f"Created directory {path.parent}")
    # plt.savefig(path)
    # cprint(f"Saved plot to {path}", "green")
    plt.close()  # Clear the current figure for the next call


# def plot_histogram_rewards(rewards, logdir, sample_iter, disc, thres):
#     print(
#         f"Plot histogram {len(rewards)=}, {logdir=}, {sample_iter=}, {disc=}, {thres=}"
#     )
#     # print("random sample of rewards", np.random.choice(rewards, 10))
#     plt.hist(rewards, bins=30, alpha=0.7, edgecolor="black")
#     plt.title(f"Histogram of predicted rewards. {sample_iter=}. {disc=}")
#     plt.xlabel("Value")
#     plt.ylabel("Frequency")
#     # plt.show()
#     path = pathlib.Path(logdir) / f"predicted_rewards_{disc=}.png"
#     if not path.parent.exists():
#         path.parent.mkdir(parents=True)
#         print(f"Create directory {path.parent}")
#     # plt.savefig(path)
#     plt.close()  # Clear the current figure for the next call
# cprint(f"Reward is >= {thres}. {disc=}. Save to {path=}", "green")


# eval driver
def per_episode(
    ep,
    env_id,
    # win_rates,
    scores,
    logger,
    args,
    mode="eval",
):
    length = len(ep["reward"])
    score = float(ep["reward"].astype(np.float64).sum())
    scores.append(score)
    logger.add(
        {
            "length": length,
            "score": score,
            "reward_rate": (ep["reward"] - ep["reward"].min() >= 0.1).mean(),
        },
        prefix=("episode" if mode == "train" else f"{mode}_episode"),
    )
    stats = {}
    for key in args.log_keys_video:
        if key in ep:
            stats[f"policy_{key}"] = ep[key]


# class DriverFineTune:
#     _CONVERSION = {
#         np.floating: np.float32,
#         np.signedinteger: np.int32,
#         np.uint8: np.uint8,
#         bool: bool,
#     }

#     def __init__(self, env, first_step=None, **kwargs):
#         # assert len(env) > 0
#         self._env = env

#         self._kwargs = kwargs
#         self._on_steps = []
#         self._on_episodes = []
#         self._first_step = first_step
#         self.policy_exclude_keys = kwargs.pop("exclude_keys", [])
#         self.reset()

#     def reset(self):
#         self._acts = {
#             k: convert(np.zeros((len(self._env),) + v.shape, v.dtype))
#             for k, v in self._env.act_space.items()
#         }
#         self._acts["reset"] = np.ones(len(self._env), bool)  # set reset = True
#         self._eps = [collections.defaultdict(list) for _ in range(len(self._env))]
#         self._state = None

#     def on_step(self, callback):
#         self._on_steps.append(callback)

#     def on_episode(self, callback):
#         self._on_episodes.append(callback)

#     def __call__(
#         self,
#         episodes=0,
#         mode="train",
#         opt_step=None,
#     ):
#         episode = 0
#         # cprint(f"{mode=}, {episodes=}", "green")
#         while episode < episodes:
#             episode = self._step(episode)

#     def _step(
#         self,
#         episode: int,
#         opt_step=None,
#     ):
#         assert all(len(x) == len(self._env) for x in self._acts.values())
#         # always acts = reset
#         acts = {k: v for k, v in self._acts.items() if not k.startswith("log_")}
#         obs = self._env.step(acts)

#         obs = {k: convert(v) for k, v in obs.items()}
#         assert all(len(x) == len(self._env) for x in obs.values()), obs

#         acts = {k: convert(v) for k, v in acts.items()}
#         # True: shape = len(self._env)
#         acts["reset"] = np.ones(len(self._env), bool)
#         self._acts = acts
#         trans = {**obs, **acts}

#         for i, first in enumerate(obs["is_first"]):
#             #     if first:
#             assert first
#             self._eps[i].clear()

#         for i in range(len(self._env)):
#             tran = {k: v[i] for k, v in trans.items()}
#             [self._eps[i][k].append(v) for k, v in tran.items()]
#             [fn(tran, i, **self._kwargs) for fn in self._on_steps]  # replay.add

#         # one step = reset all envs
#         episode += len(self._env)

#         return episode

#     def _expand(self, value, dims):
#         while len(value.shape) < dims:
#             value = value[..., None]
#         return value


def plot_stds_for_trial_numbers(
    k2stds, game_iter, hard_games, NUM_TRIALS, NUM_K, config
):
    # plot
    plt.figure(figsize=(10, 6))
    values = [np.sqrt(np.sum(np.array(v) ** 2) / len(v)) for v in k2stds.values()]
    plt.plot(list(range(1, NUM_K)), values, marker="o", color="blue", linestyle="-")
    plt.xlabel("Index")
    plt.ylabel("Value")
    plt.title(f"std of {game_iter=}, {len(hard_games)=}, {NUM_TRIALS=}")
    plt.grid(True)
    checkpoint = pathlib.Path(config.run.from_checkpoint).parent.name
    file_path = pathlib.Path(checkpoint) / "std.png"
    if not file_path.parent.exists():
        file_path.parent.mkdir(parents=True)
        print(f"Create directory {file_path.parent}")
    # plt.savefig(file_path)
    # cprint(f"Save plot to {file_path}", "green")
    if config.use_wandb:
        fig = plt.gcf()
        if wandb.run is not None:
            wandb.log({"std_for_trials": wandb.Image(fig)})
    plt.close()


def find_trial_number(
    agent: "Agent",
    eval_envs,
    driver_real: "Driver",
    before_real_scores: List[float],
    config,
    skip_successful=False,
):
    print(f"Start training loop. {skip_successful=}")
    old_agent = agent.save()
    hard_games = []

    NUM_GAMES = 100
    NUM_TRIALS = 100
    NUM_K = len(eval_envs)
    NUM_ENV_HARD = 20

    while len(hard_games) < NUM_GAMES:
        game_config = eval_envs.create_game_config()
        eval_envs.reset_game_config(**game_config)

        agent.load(old_agent)
        before_real_scores.clear()
        driver_real.reset()
        with Timing("Rollout this game config"):
            driver_real.__call__(
                partial(test_policy, agent),
                max_episodes=NUM_ENV_HARD,
            )
        assert len(before_real_scores) == NUM_ENV_HARD, (
            len(before_real_scores),
            NUM_ENV_HARD,
        )

        MAX_SCORE = {
            "s1": 1,
            "s2": 1.5,
            "s3": 1.5,
            "lwm_easy": 1.5,
            "lwm_medium": 1.5,
            "lwm_hard": 1.5,
        }[config.task]
        is_hard_game = np.mean(before_real_scores) < MAX_SCORE
        if is_hard_game:
            hard_games.append(game_config)
            cprint(f"Find {len(hard_games)} hard games.", "green")

    cprint(f"Find {len(hard_games)} hard games.", "green")
    from collections import defaultdict

    k2stds = defaultdict(list)
    cprint(f"Setting {len(hard_games)} hard games.", "green")
    BIG_ENOUGH_EPISODES = 10000

    for _, hard_game in tqdm(enumerate(hard_games), desc="Hard games"):
        eval_envs.reset_game_config(**hard_game)
        mean_scores = []
        # print(f"Number of trials: {k}")
        agent.load(old_agent)
        before_real_scores.clear()
        driver_real.reset()

        with Timing(f"rollout policy in {BIG_ENOUGH_EPISODES} episodes"):
            driver_real.__call__(
                partial(test_policy, agent),
                max_episodes=BIG_ENOUGH_EPISODES,
            )
        assert len(before_real_scores) == BIG_ENOUGH_EPISODES, (
            len(before_real_scores),
            BIG_ENOUGH_EPISODES,
        )

        for k in tqdm(range(1, NUM_K), desc="Test each k"):
            before_trial_scores = np.random.choice(before_real_scores, (NUM_TRIALS, k))
            mean_scores = np.mean(before_trial_scores, axis=1)  # NUM_TRIALS,
            std_score = np.std(mean_scores)
            k2stds[k].append(std_score)

    # pooled std
    # for k, v in k2stds.items():
    #     print(k, np.sqrt(np.sum(np.array(v) ** 2) / len(v)))

    plot_stds_for_trial_numbers(
        k2stds,
        game_iter=NUM_GAMES,
        hard_games=hard_games,
        NUM_TRIALS=NUM_TRIALS,
        NUM_K=NUM_K,
        config=config,
    )


def avg_imag_return(agent: "Agent", batch):
    mets = agent.report_policy(batch)
    return avg_discounted_return(
        mets["reward"], mets["cont"], agent.config.report_return_lambda
    )


# def inspect_rewards_pred(agent, batch):
#     mets = agent.report_policy(batch)
#     plot_hist_rewards_pred(mets["reward"], mets["cont"])


def plot_hist_rewards_pred(reward_preds):
    # remove all value within -0.05 and 0.05
    # keep all values < -0.05 and > 0.05
    reward_preds = reward_preds[(reward_preds < -0.05) | (reward_preds > 0.05)]

    # save fig to wandb
    def hist(reward_preds, tag):
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(reward_preds, bins=30, alpha=0.7, edgecolor="black")
        ax.set_title("Histogram of predicted rewards")
        ax.set_xlabel("Value")
        ax.set_ylabel("Frequency")
        # plt.show()
        fig = plt.gcf()
        # if wandb is running
        if wandb.run is not None:
            wandb.log({tag: wandb.Image(fig)})
        plt.close()

    hist(reward_preds, "hist_reward_pred_whole")


def plot_rewards_pred_freq(rewards, in_eps):
    # Reshape from (horizon+1, bs) to (bs, horizon+1)
    rewards = rewards.T
    in_eps = in_eps.T
    assert rewards.shape == in_eps.shape, (rewards.shape, in_eps.shape)

    # Define reward intervals
    intervals = [
        (0.9, 1.1),
        (0.8, 0.9),
        (0.7, 0.8),
        (0.6, 0.7),
        (0.5, 0.6),
        (0.4, 0.5),
        (0.3, 0.4),
        (0.2, 0.3),
        (0.1, 0.2),
        (0.05, 0.1),
        (-0.1, -0.05),
        (-0.2, -0.1),
        (-0.3, -0.2),
        (-0.4, -0.3),
        (-0.5, -0.4),
        (-0.6, -0.5),
        (-0.7, -0.6),
        (-0.8, -0.7),
        (-0.9, -0.8),
        (-1.1, -0.9),
    ]
    # reverse the list
    interval_labels = intervals[::-1]
    interval_labels = [f"{low}-{high}" for low, high in intervals]

    # Calculate the frequency of rewards within each interval for each episode
    interval_counts = np.zeros((rewards.shape[0], len(intervals)))
    for i, (low, high) in enumerate(intervals):
        interval_counts[:, i] = np.sum(
            (rewards >= low) & (rewards < high) & (in_eps == 1), axis=1
        )

    # Calculate the average frequency of each interval across episodes
    avg_interval_counts = np.mean(interval_counts, axis=0)

    # Plot the histogram of these frequencies
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(
        interval_labels, avg_interval_counts, color="blue", alpha=0.7, edgecolor="black"
    )
    ax.set_title("Histogram of Reward Frequencies within Episodes")
    ax.set_xlabel("Reward Intervals")
    ax.set_ylabel("Average Frequency")
    plt.xticks(rotation=90)
    plt.tight_layout()
    # plt.show()
    fig = plt.gcf()
    if wandb.run is not None:
        wandb.log({"hist_reward_freq": wandb.Image(fig)})
    plt.close()


# Example usage
def test_plot_rewards_pred_freq():
    rewards = np.random.uniform(-1, 1, (6, 5))
    cont = np.random.choice([0, 1], size=(6, 5), p=[0.8, 0.2])
    in_eps = compute_in_eps(cont)
    print(rewards)
    print(in_eps)
    plot_rewards_pred_freq(rewards, in_eps)


#     cont_zero_mask = mets["cont"] == 0  # mets['cont']: bl, bs
#     first_zero_index = np.argmax(cont_zero_mask, axis=0)
#     # replace 0 with the last index
#     first_zero_index = np.where(
#         first_zero_index == 0, mets["cont"].shape[0] - 1, first_zero_index
#     )
#     # assert first_zero_index doesn't have zeros
#     assert (first_zero_index == 0).sum() == 0, first_zero_index
#     first_reward_when_cont_zero = mets["reward"][
#         first_zero_index, np.arange(mets["reward"].shape[1])
#     ]
#     return_sums_in_eps = (mets["reward"] * mets["cont"]).sum(
#         axis=0
#     ) + first_reward_when_cont_zero
#     return return_sums_in_eps


def compute_first_obsss(eval_envs):
    # get the first obsss
    reset_act = {
        k: convert(np.zeros((len(eval_envs),) + v.shape, v.dtype))
        for k, v in eval_envs.act_space.items()
    }
    reset_act["reset"] = np.ones(len(eval_envs), bool)  # set reset = True
    first_obsss = eval_envs.step(reset_act)
    first_obsss = {**first_obsss, **reset_act}
    first_obsss = {k: convert(v) for k, v in first_obsss.items()}  # bs, d*
    first_obsss = {
        k: v for k, v in first_obsss.items() if k not in IGNORE_KEYS and "log" not in k
    }
    first_obsss = {
        k: np.expand_dims(v, axis=1) for k, v in first_obsss.items()
    }  # bs, 1, d*
    return first_obsss


def finetune_policy_on_fixed_wm(
    agent: "Agent",
    eval_envs,
    logger,
    config,
    lose_only=False,
    skip_from_wm=False,
):
    THRESHOLD_WIN = int(len(eval_envs) * 3 / 4)
    cprint(f"{THRESHOLD_WIN=}")
    args = config.run
    old_agent = agent.save()
    worse_win_rate_cnt = 0
    better_win_rate_cnt = 0
    worse_mean_cnt = 0
    better_mean_cnt = 0
    losses = 0
    cnt_eps_finetune = 0
    NUM_ENV_TEST = config.num_eval_eps
    after_imag_diffs = []
    after_real_diffs = []
    before_real_returns_ft = []  # for finetune scores

    before_real_returns = []  # mean for each game
    after_real_returns = []  # mean for each game
    before_eps_scores: List[List[float]] = []
    after_eps_scores: List[List[float]] = []
    before_imag_returns = []
    diffs = []

    driver_real = Driver(eval_envs)
    before_real_scores_per_game = []
    after_real_scores_per_game = []
    replay = make_replay(
        config,
        None,
        True,
        rate_limit=False,
    )

    # replay = None
    def add_to_replay(replay, *args):
        if replay is not None:
            replay.add(*args)

    driver_real.on_step(partial(add_to_replay, replay))

    driver_real.on_episode(
        lambda ep, env_id: per_episode(
            ep,
            env_id,
            before_real_scores_per_game,
            logger=logger,
            args=args,
            mode="eval",
        )
    )

    driver_after = Driver(eval_envs)
    driver_after.on_episode(
        lambda ep, env_id: per_episode(
            ep,
            env_id,
            # after_win_rates,
            after_real_scores_per_game,
            logger=logger,
            args=args,
            mode="finetune",
        )
    )

    for game_iter in tqdm(range(args.eps_finetune), desc="Finetune each eps"):
        game_config = eval_envs.create_game_config()
        eval_envs.reset_game_config(**game_config)

        # run agent policy in real envs
        agent.load(old_agent)
        before_real_scores_per_game.clear()
        if replay is not None:
            replay.reset()

        driver_real.reset()
        driver_real.__call__(
            partial(test_policy, agent),
            max_episodes=NUM_ENV_TEST,
        )
        assert len(before_real_scores_per_game) == NUM_ENV_TEST, (
            len(before_real_scores_per_game),
            NUM_ENV_TEST,
        )

        # this_game_mostly_win = before_win_rates.count(True) >= THRESHOLD_WIN
        # this_game_mostly_lost = before_win_rates.count(False) >= THRESHOLD_WIN
        # assert any value in before_real_scores in [-1, 1] if task = s1
        if "s1" in config.task:
            assert all(e in [-1, 1] for e in before_real_scores_per_game), (
                before_real_scores_per_game
            )
            assert all(
                e == before_real_scores_per_game[0] for e in before_real_scores_per_game
            ), set(before_real_scores_per_game)

        before_real_return = np.mean(before_real_scores_per_game)
        if lose_only:
            WIN_SCORES = {
                "messenger_s1": 0.9,
                "messenger_s2": 1.4,
                "messenger_s3": 1.4,
                "lwm_easy": 1.4,
                "lwm_medium": 1.4,
                "lwm_hard": 1.4,
            }
            if before_real_return >= WIN_SCORES[config.task]:
                cprint(f"{before_real_return=}, SKIP!")
                continue

        before_real_returns.append(before_real_return)
        before_eps_scores.append(before_real_scores_per_game)

        first_obsss = compute_first_obsss(eval_envs)
        batch_first_obsss = agent.postprocess(first_obsss)  # bs, 1, d*
        before_imag_return = avg_imag_return(agent, batch_first_obsss)
        before_imag_returns.append(before_imag_return)

        plot_scatter_before_means_real_vs_imag(before_real_returns, before_imag_returns)
        # plot value gap under this policy
        plot_value_gap(
            before_real_returns,
            before_imag_returns,
            checkpoint=pathlib.Path(config.run.from_checkpoint).parent.name,
            num_eps=game_iter,
            config=config,
            step_finetune=config.run.step_finetune,
        )
        before_real_returns_mean = np.mean(before_real_returns)
        print(f"{game_iter=}, real_returns_mean= {before_real_returns_mean}")

        # condition_wm = "reward < 0.9"
        if skip_from_wm:
            does_skip = {
                "messenger_s1": before_imag_return > 0.9,
                "messenger_s2": before_imag_return > 1.2,
                "messenger_s3": before_imag_return > 1.4,
                "lwm_easy": before_imag_return > 1.4,
                "lwm_medium": before_imag_return > 1.4,
                "lwm_hard": before_imag_return > 1.4,
            }
            if does_skip[config.task]:
                cprint(
                    f"Skip this game {before_imag_return=}, {before_real_return=}",
                    "yellow",
                )
                after_real_returns.append(before_real_return)
                after_eps_scores.append(before_real_scores_per_game)
                continue
            else:
                cprint(
                    f"Need to turn this game {before_imag_return=}, {before_real_return=}",
                    "yellow",
                )

        cnt_eps_finetune += 1
        for _ in tqdm(range(int(args.step_finetune)), desc="Grad step"):
            _, _, mets = agent.finetune_policy(
                batch_first_obsss
                # uncertainty=uncertainty,
            )

        # bs, bl * 10, d* -> bs, bl, d*
        after_imag_return = avg_imag_return(agent, batch_first_obsss).mean()
        imag_diff = after_imag_return - before_imag_return
        MAX_GRAD_STEPS = 10000
        grad_step_sofar = args.step_finetune
        not_improve_best = 0
        max_imag_diff = 0
        MAX_IMAG_IMPROVEMENT = 1
        MAX_PATIENCE_NOT_IMPROVE = 1
        best_agent_during_ft = None
        # best_agent = None

        print(f"{imag_diff=}, {grad_step_sofar=}, {not_improve_best=}")
        while (
            imag_diff < MAX_IMAG_IMPROVEMENT
            and grad_step_sofar < MAX_GRAD_STEPS
            and not_improve_best < MAX_PATIENCE_NOT_IMPROVE  # 2000 grad steps
        ):
            grad_step_sofar += args.step_finetune
            cprint(f"Train more. {imag_diff=}, {max_imag_diff=}, {not_improve_best=}")
            for _ in tqdm(range(args.step_finetune), desc="Grad step"):
                _, _, mets = agent.finetune_policy(batch_first_obsss)

            after_imag_return = avg_imag_return(agent, batch_first_obsss)
            imag_diff = after_imag_return - before_imag_return
            not_improve_best += imag_diff - max_imag_diff < 0.05 or imag_diff < 0
            print(f"{imag_diff=}, {max_imag_diff=}")

            if imag_diff > max_imag_diff:
                if abs(imag_diff - max_imag_diff) >= 0.05 and imag_diff > 0:
                    not_improve_best = 0

                max_imag_diff = imag_diff
                best_agent_during_ft = agent.save()

        if (
            max_imag_diff > config.imag_improve_thres
            and max_imag_diff < config.imag_improve_max
        ):
            cprint(f"Load best agent at {max_imag_diff=}", "green")
            assert best_agent_during_ft is not None
            agent.load(best_agent_during_ft)
            after_imag_diffs.append(max_imag_diff)

            # evaluate policy
            after_real_scores_per_game.clear()
            driver_after.reset()
            driver_after.__call__(
                partial(test_policy, agent),
                max_episodes=NUM_ENV_TEST,
            )
            assert len(after_real_scores_per_game) == NUM_ENV_TEST, len(
                after_real_scores_per_game
            )
            # after_win = after_win_rates.count(True) >= THRESHOLD_WIN_LOST
            after_real_return = np.mean(after_real_scores_per_game)
            after_real_returns.append(after_real_return)
            after_eps_scores.append(after_real_scores_per_game)
        else:
            cprint(f"{max_imag_diff=} < {config.imag_improve_thres}")
            cprint("load old agent", "red")
            agent.load(old_agent)
            after_imag_diffs.append(0)
            after_real_returns.append(before_real_returns[-1])

        if before_real_returns[-1] > after_real_returns[-1]:
            cprint(
                f"GET WORSE! Before mean: {before_real_returns[-1]}, After mean: {after_real_returns[-1]}",
                "red",
            )
            worse_mean_cnt += 1

        elif before_real_returns[-1] < after_real_returns[-1]:
            better_mean_cnt += 1
            cprint(
                f"Improved! Before mean: {before_real_returns[-1]}, After mean: {after_real_returns[-1]}",
                "green",
            )

        else:
            cprint(f"No change! {before_real_returns[-1]=}, {after_real_returns[-1]=}.")

        after_real_diffs.append(after_real_returns[-1] - before_real_returns[-1])
        before_real_returns_ft.append(before_real_returns[-1])
        assert len(before_real_returns) == len(after_real_returns), (
            len(before_real_returns),
            len(after_real_returns),
        )

        cprint(
            f"{game_iter=}, {losses=}, {better_mean_cnt=}, {np.mean(before_real_returns)=}, {worse_mean_cnt=}, {np.mean(after_real_returns)=}",
            "green",
        )
        BIG_ENOUGH_FOR_TEST = 30
        diff = after_real_returns[-1] - before_real_returns[-1]
        if diff != 0:
            diffs.append(diff)
            if len(diffs) > BIG_ENOUGH_FOR_TEST:
                p_test(before_real_returns, after_real_returns)
                bootstrap(before_real_returns, after_real_returns)
                if len(diffs) % 50 == 0:
                    hier_bootstrap(before_eps_scores, after_eps_scores)
            else:
                cprint(f"{len(diffs)} samples is not big enough", "yellow")

        MAX_SCORE = {
            "messenger_s1": 1,
            "messenger_s2": 1.5,
            "messenger_s3": 1.5,
            "lwm_easy": 1.5,
            "lwm_medium": 1.5,
            "lwm_hard": 1.5,
        }[config.task]
        MIN_SCORE = -1
        # scale this to 0 to 1. given MIN_SCORE and MAX_SCORE
        # win_levels = (np.array(before_real_scores) - MIN_SCORE) / (
        #     MAX_SCORE - MIN_SCORE
        # )
        # do list instead

        win_levels = [
            (e - MIN_SCORE) / (MAX_SCORE - MIN_SCORE) for e in before_real_returns_ft
        ]
        plot_scatter_after_imag_real_diff_ft(
            after_imag_diffs,
            after_real_diffs,
            win_levels,
            checkpoint=pathlib.Path(config.run.from_checkpoint).parent.name,
            num_eps=cnt_eps_finetune,
            step_finetune=config.run.step_finetune,
        )

    cprint(f"Finetune on {args.eps_finetune} episodes.", "green")
    after_mean_score = np.mean(after_real_returns)
    before_mean_score = np.mean(before_real_returns)
    cprint(f"{after_mean_score=}, {before_mean_score=}", "green")
    cprint(f"{losses=}, {better_win_rate_cnt=}, {worse_win_rate_cnt=}", "green")
    if wandb.run is not None:
        wandb.log(
            {
                "after_diff_means": np.mean(after_real_returns)
                - np.mean(before_real_returns)
            }
        )


def compute_on_fixed_wm(
    agent: "Agent",
    eval_envs,
    logger: "Logger",
    config,
):
    args = config.run
    logdir = embodied.Path(config.logdir)
    logdir.mkdirs()
    print("Logdir", logdir)

    should_save = embodied.when.Clock(args.save_every)
    step = logger.step
    real_env_step = embodied.Counter()
    metrics = embodied.Metrics()

    print("Observation space:")
    for key, value in eval_envs.obs_space.items():
        print(f"  {key:<16} {value}")
    print("Action space:")
    for key, value in eval_envs.act_space.items():
        print(f"  {key:<16} {value}")

    timer = Timer()
    timer.wrap("agent", agent, ["policy", "train", "report", "save"])
    timer.wrap("env", eval_envs, ["step"])
    timer.wrap("logger", logger, ["write"])

    # driver_eval.on_step(lambda *args: num_eps.increment())
    logger.add(metrics.result())
    logger.write()
    # state = [None]  # To be writable from train step function below.

    checkpoint = embodied.Checkpoint(logdir / "checkpoint_1.ckpt", not args.overfit_eps)
    timer.wrap("checkpoint", checkpoint, ["save", "load"])
    checkpoint.step = step
    checkpoint.real_step = real_env_step
    checkpoint.agent = agent
    if args.from_checkpoint:
        checkpoint.load(args.from_checkpoint)
    checkpoint.save()

    # if config.finetune_script == "find_trial_number":
    #     find_trial_number(
    #         agent,
    #         eval_envs,
    #         driver_real,
    #         # before_win_rates,
    #         before_real_scores_per_eps,
    #         config,
    #         skip_successful=False,
    #     )

    if config.finetune_script == "run_finetune":
        finetune_policy_on_fixed_wm(
            agent,
            eval_envs,
            logger,
            config,
            lose_only=config.lose_only,
            # skip_win=config.skip_win,
            skip_from_wm=config.skip_from_wm,
        )

    # def tune_finetune():
    #     for critic_lr in np.arange(5e-4, 1e-6, -1e-5):
    #         for grad_step in np.arange(1000, 10000, 500):
    #             cprint(f"Start training loop. {critic_lr=}, {grad_step=}", "green")
    #             config = config.update({"critic_opt": {"lr": critic_lr}})
    #             run_finetune(agent, skip_win=False, skip_from_wm=True)

    # tune_finetune()

    if config.finetune_script == "measure_return":
        measure_return(
            agent,
            eval_envs,
            logger,
            checkpoint,
            config,
        )


def compute_in_eps(cont):  # horizon+1, bs
    # Create a mask where cont is 1
    in_eps = np.zeros_like(cont, dtype=int)
    in_eps[cont == 1] = 1

    # Find the first occurrence of cont being 0 for each episode
    # print(cont)
    first_zero_index = np.argmax(cont == 0, axis=0)  #
    last_time_step = cont.shape[0] - 1
    # print(first_zero_index)
    first_zero_index = np.where(first_zero_index == 0, last_time_step, first_zero_index)
    # print(first_zero_index)

    # Set in_eps to 1 for all steps before the first occurrence of cont being 0
    for sample in range(cont.shape[1]):
        in_eps[: first_zero_index[sample] + 1, sample] = 1
        in_eps[first_zero_index[sample] + 1 :, sample] = 0

    return in_eps


def test_in_eps():
    cont = np.array(
        [
            [1, 1, 1, 1, 1],
            [1, 0, 1, 1, 1],
            [1, 0, 0, 1, 1],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0],
        ]
    )

    in_eps = compute_in_eps(cont)
    print(in_eps)


def test():
    # test_compute_discounted_reward_sum()
    # test_in_eps()
    test_plot_rewards_pred_freq()


if __name__ == "__main__":
    test()
