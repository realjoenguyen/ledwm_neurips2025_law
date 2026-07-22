from collections import defaultdict
import copy
import threading
from scipy.ndimage import gaussian_filter1d
import numpy as np
from termcolor import cprint
from ledwm.embodied.core.config import Config
from ledwm.embodied.replay.generic import GenericReplay


from ledwm.jaxutils import get_task


# Wrapper to apply gaussian smoothing before adding to replay buffer
# current strategy: since episode lengths not assumed to be uniform, we will have a buffer to store the current episode
# only when episode completes, will we process and store entire episode's transitions to the replay buffer

CNT_ERROR = 0
POS_REWARD_INDICATOR = {
    "last": {
        "messenger_s1": 1,
        "messenger_s2": 1,
        "messenger_s3": 1,
        "lwm_easy": 1,
        "lwm_hard": 1,
        "lwm_medium": 1,
    },
    "sum": {
        "messenger_s1": 1,
        "messenger_s2": 1.5,
        "messenger_s3": 1.5,
        "lwm_easy": 1.5,
        "lwm_hard": 1.5,
        "lwm_medium": 1.5,
    },
}


def apply_left_gaussian(rewards, sigma=2):
    rewards = np.array(rewards).astype(float)
    res = np.copy(rewards)

    last_nonzero = -1

    for i in range(len(rewards)):
        if rewards[i] != 0:
            left_segment = rewards[last_nonzero + 1 : i + 1]
            # print(left_segment)
            if len(left_segment) == 1:
                # add zero to the left segment
                left_segment = np.append(0, left_segment)
                add_zero = True
            else:
                add_zero = False

            if len(left_segment) > 0:
                filtered_segment = gaussian_filter1d(
                    left_segment, sigma=sigma, mode="nearest"
                )
                if add_zero:
                    filtered_segment = filtered_segment[1:]

                # print("filtered_segment", np.round(filtered_segment, 2))
                res[last_nonzero + 1 : i + 1] = filtered_segment
            last_nonzero = i

    return np.round(res, 2)


def is_empty_eps_stream(eps_stream):
    assert eps_stream[0]["is_first"], eps_stream[0]
    return np.all(eps_stream[0]["entity_pos"] == 0)


class ReplayEps:
    def __init__(
        self,
        replay,
        sigma=0,
        zero_first=True,
        config: "Config" = None,
    ):
        self._replay: "GenericReplay" = replay
        self.reward_buffer = defaultdict(list)
        self.current_eps_trans = defaultdict(list)
        self.sigma = float(sigma)
        self.zero_first = zero_first
        # assert self.sigma > 0, "make sure gaussian smooothing sigma > 0"
        self.config = config
        self.lock = threading.Lock()
        self.count_error = 0

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        try:
            return getattr(self._replay, name)
        except AttributeError:
            raise ValueError(name)

    def reset(self):
        self.reward_buffer.clear()
        self.current_eps_trans.clear()
        self._replay.reset()

    def set_train_ratio(self, train_ratio):
        self._replay.set_train_ratio(train_ratio)

    def save(self, wait=False):
        self._replay.save(wait=wait)

    def __len__(self):
        return len(self._replay)

    def __getitem__(self, idx):
        return self._replay[idx]

    def add(self, step, worker=0, load=False, training=True):
        step = step.copy()
        step.update({"kwargs": {"load": load}})
        eps_stream = self.current_eps_trans[worker]
        eps_reward_buffer = self.reward_buffer[worker]
        if step["is_first"]:
            # TODO : can't debug this
            if len(eps_stream) > 0:
                self.count_error += 1
                cprint(
                    f"ERROR step['is_first'] in not empty stream : count = {self.count_error}",
                    "red",
                )
                eps_stream.clear()
                eps_reward_buffer.clear()

            assert len(eps_stream) == 0, eps_stream
        else:
            # TODO can't debug
            try:
                assert len(eps_stream) > 0, len(eps_stream)
            except AssertionError as e:
                self.count_error += 1
                cprint(
                    f"ERROR step['is_first'] in not empty stream : count = {self.count_error}",
                    "red",
                )

        with self.lock:
            eps_stream.append(step)
            eps_reward_buffer.append(step["reward"])

        if step["is_last"]:
            if get_task(self.config) == "s3":
                assert eps_reward_buffer[-1] in [
                    -1,
                    1,
                    -2,
                    -3,
                ], f"{eps_reward_buffer[-1]=}"
            else:
                assert eps_reward_buffer[-1] in [-1, 1], f"{step=}"

            if not eps_stream[0]["is_first"]:
                cprint("replay.invalid_episode_start | source=smoothing", "red")
                cprint("replay.invalid_episode_start | source=smoothing", "red")
                cprint("replay.invalid_episode_start | source=smoothing", "red")

            # assert eps_stream[0]["is_first"], eps_stream[0]
            assert eps_stream[-1]["is_last"], eps_stream[-1]

            with self.lock:
                if self.config.ignore_error:
                    if eps_reward_buffer[-1] != -2:
                        self._add_eps_to_replay(worker, training=training)
                else:
                    self._add_eps_to_replay(worker, training=training)

            with self.lock:
                eps_reward_buffer.clear()
                eps_stream.clear()

    def _add_eps_to_replay(
        self,
        worker,
        eps_stream=None,
        reward_stream=None,
        training=True,
        aug=False,
    ):
        # do gaussian smoothing, then add all steps to replay buffer
        if eps_stream is None:
            assert reward_stream is None
            eps_stream = self.current_eps_trans[worker]
            reward_stream = self.reward_buffer[worker]

        assert reward_stream is not None
        assert eps_stream is not None
        # assert reward_stream[-1] in [-1, 1], reward_stream[-1]

        assert self.config.replay.imbalance_reward in [
            "last",
            "sum",
        ], self.config.replay.imbalance_reward
        if self.config.replay.imbalance_reward == "last":
            reward_indicator = reward_stream[-1]
        else:
            reward_indicator = sum(reward_stream)

        if self.sigma > 0:
            rew_smooth = apply_left_gaussian(reward_stream, self.sigma)
            if self.zero_first:
                rew_smooth[0] = 0
            assert len(rew_smooth) == len(reward_stream) == len(eps_stream), (
                f"{len(rew_smooth)=} == {len(reward_stream)=} == {len(eps_stream)=}"
            )

        def add_eps(eps_stream, reward_indicator):
            for i, step in enumerate(eps_stream):
                if "kwargs" not in step:
                    cprint("replay.step_missing_kwargs | source=smoothing", "red")
                kwargs = step.pop("kwargs", {})

                if self.sigma > 0:
                    assert len(rew_smooth) == len(eps_stream), (
                        f"{len(rew_smooth)=}, {len(eps_stream)=}"
                    )
                    step.update({"reward": rew_smooth[i]})
                if self.config.replay.type == "curious":
                    kwargs = {}

                kwargs["training"] = training
                self._replay.add(
                    step,
                    worker,
                    aug=aug,
                    reward_indicator=reward_indicator,
                    # is_read_step=step["is_read_step"],
                    **kwargs,
                )

        eps_stream_dup = copy.deepcopy(eps_stream)
        if is_empty_eps_stream(eps_stream):
            cprint("replay.empty_episode_stream | source=smoothing", "red")
            return

        add_eps(eps_stream, reward_indicator)
        # if replay doesn't have positive - reward indicator == 1.5 then duplicate this episode

        min_pos_eps = 1 if self.config.replay.imbalance == "upsample_pos" else 5
        if self.config.duplicate_pos_eps:
            is_pos_reward_indicator = (
                reward_indicator
                == POS_REWARD_INDICATOR[self.config.replay.imbalance_reward][
                    self.config.task
                ]
            )
            if (
                is_pos_reward_indicator
                and self._replay.counts_eps_reward[reward_indicator] < min_pos_eps
                and self._replay.is_train
            ):
                for _ in range(4):
                    add_eps(copy.deepcopy(eps_stream_dup), reward_indicator)
                cprint(
                    f"duplicating episode with {reward_indicator=}. Counts_eps_reward={self._replay.counts_eps_reward[reward_indicator]}, {self._replay.is_train=}",
                    "red",
                )
