from copy import deepcopy
import threading
import time
from collections import defaultdict, deque
from functools import partial as bind
from typing import TYPE_CHECKING, OrderedDict
from termcolor import cprint
from tqdm import tqdm
import numpy as np

from ledwm.embodied.core.basics import convert
from ledwm.embodied.core.uuid import uuid
from ledwm.embodied.replay.saver import REPLAY_STEP_ID_KEY
from ledwm.logging_setup import logger as event_logger

# get_task (jaxutils) imported lazily in GenericReplay.__init__

if TYPE_CHECKING:
    from ledwm.embodied.replay.Prioritized import PrioritizedSampler
    from ledwm.embodied.replay.selectors import Fifo, Uniform

KEEP_KEYS = None
EPS = 1e-6
MAX_BALANCED_WEIGHT = 3
REPLAY_SAMPLE_COUNT_KEY = "_replay_sample_count"
REPLAY_SAMPLE_PRIORITY_KEY = "_replay_sample_priority"


def _canonical_reward_indicator(value):
    """Match two-decimal replay bucketing without NumPy scalar dispatch."""
    value = np.float32(value)
    return np.float32(
        np.rint(value * np.float32(100.0)) / np.float32(100.0)
    )


def consume_replay_sample_stats(batch):
    """Remove and summarize telemetry captured for this exact replay batch."""
    counts = batch.pop(REPLAY_SAMPLE_COUNT_KEY, None)
    priorities = batch.pop(REPLAY_SAMPLE_PRIORITY_KEY, None)
    mean_count = float(np.asarray(counts).mean()) if counts is not None else np.nan
    mean_priority = (
        float(np.asarray(priorities).mean()) if priorities is not None else np.nan
    )
    return mean_count, mean_priority


def convert2uuid(keys):
    return [uuid(k.astype(np.uint8)) for k in keys]


class ReplayProgress:
    """Small checkpoint entry for replay sampling debt, not replay payloads."""

    def __init__(self, replay):
        while hasattr(replay, "_replay"):
            replay = replay._replay
        self.replay = replay

    def save(self):
        return self.replay.save_progress()

    def load(self, state):
        self.replay.load_progress(state)


def wait(predicate, message, sleep=0.001, notify=1.0):
    first = True
    start = time.time()
    notified = False
    while True:
        allowed, detail = predicate()
        duration = time.time() - start
        # if duration is > 30 minutes, timeout and return 0
        if duration > 1800:
            raise TimeoutError(f"Timeout waiting for {message}")
        if allowed:
            return 0 if first else duration
        if not notified and duration >= notify:
            print(f"{message} | detail={detail}")
            notified = True
        time.sleep(sleep)
        first = False


class GenericReplay:
    def __init__(
        self,
        batch_length,  # batch_length
        capacity,  # config.replay.size
        remover: "Fifo",
        sampler: "Uniform | PrioritizedSampler",
        limiter,
        directory,  # logdir / "episodes"
        train_ratio,
        min_size=1,
        overlap=None,
        chunks=1024,
        load_directories=None,  # None for parallel
        dataset_zero_keys=None,
        is_eval=False,
        online=False,
        config={},
        verbose_every=50,
    ):
        assert capacity is None or 1 <= capacity
        self.verbose_every = verbose_every
        self.finetune_policy = config.run.script == "finetune_policy"
        self.done_load = False
        self.error_cnt_is_first = 0
        self.min_size = min_size
        self._fill_progress = None
        self._fill_progress_done = False
        self._fill_progress_lock = threading.Lock()
        self.config = config
        from ledwm.jaxutils import get_task

        self.task = get_task(config)
        self.is_eval = is_eval
        self.overfit_eps = config.run.overfit_eps
        self.upsample_pos = config.replay.imbalance == "upsample_pos" or is_eval
        self.train_ratio = train_ratio
        assert self.train_ratio > 0, self.train_ratio
        if self.upsample_pos:
            self.upsample_pos_rate = config.replay.upsample_pos_rate
            cprint(
                f"replay.positive_upsampling | enabled=true | "
                f"is_eval={str(is_eval).lower()} | "
                f"rate={self.upsample_pos_rate}",
                "green",
            )

        self.is_first = config.replay.is_first
        self.batch_length = batch_length
        self.size = capacity
        self.remover: "Fifo" = remover
        self.sampler = sampler
        if self.upsample_pos:
            self.remover_pos = deepcopy(remover)
            self.sampler_pos = deepcopy(sampler)

        self.limiter = limiter
        self.stride = 1 if overlap is None else batch_length - overlap
        # self.streams = defaultdict(bind(deque, maxlen=length))
        # stream: 2D defaultdict -> defauldict of bind(...)
        # if self.seperate_reward_stream:
        self.streams = defaultdict(
            lambda: defaultdict(bind(deque, maxlen=batch_length))
        )
        self.counts_insert_step = defaultdict(lambda: defaultdict(int))
        self.counters = defaultdict(lambda: defaultdict(int))

        self.counts_stream2sample = defaultdict(int)
        self.counts_eps_reward = defaultdict(int)
        self.counts_eps = defaultdict(int)
        self.counts_stream_reward = defaultdict(int)

        # key = uuid
        # each is the seq (bl)
        # self.table = {}
        # self.table_pos = {}
        self.table = OrderedDict()
        self.table_pos = OrderedDict()
        # Sampling used to rebuild every sequence from its tuple of step dicts.
        # Prioritized replay samples the same sequence many times, so materialize
        # it once at insertion and reuse the arrays until the sequence is removed.
        self._materialized = {}
        self._materialized_nbytes = 0

        self.lock = threading.Lock()
        self.finetune_online = online
        self.preloaded = {}
        self.chunks = chunks
        if self.finetune_online:
            self.online_queue = deque()
            self.online_stride = batch_length
            # self.online_counters = defaultdict(int)
            # 2d defaultdict instead
            self.online_counters = defaultdict(lambda: defaultdict(int))

        self.itemsize = 0
        self.metrics = {
            "samples": 0,
            "sample_wait_dur": 0,
            "sample_wait_count": 0,
            "inserts": 0,
            "insert_wait_dur": 0,
            "insert_wait_count": 0,
        }

        # Initialize dataset transforms before loading. Loading feeds saved
        # steps through add(), which now materializes each completed sequence.
        dataset_exclude_keys = config.dataset_exclude_keys
        self.dataset_exclude_keys = (
            set(dataset_exclude_keys) if dataset_exclude_keys is not None else []
        )
        print(f"replay.dataset_filter | excluded_keys={self.dataset_exclude_keys}")
        self._dataset_zero_keys = (
            set(dataset_zero_keys) if dataset_zero_keys is not None else []
        )
        print(f"replay.dataset_filter | zero_keys={self._dataset_zero_keys}")

        if self.is_train:
            if load_directories:
                cprint(
                    f"replay.load | source={load_directories} | "
                    f"save_directory={directory}",
                    "green",
                )
                for load_dir in load_directories:
                    self.preload_from_dir(load_dir)

                # Preloaded sequences are immediately valid learner data. Without
                # this credit, SamplesPerInsert starts at its negative limit and
                # waits for fresh actor inserts despite a full replay on disk.
                if len(self) and hasattr(self.limiter, "set_avail_max"):
                    self.limiter.set_avail_max()

                # Even if we load from a different directory, save to this expdir
                from . import saver

                self.saver = directory and saver.Saver(
                    directory, chunks, capacity, batch_length
                )
            else:
                from . import saver

                self.saver = directory and saver.Saver(
                    directory, chunks, capacity, batch_length
                )
                if config.replay.resume:
                    cprint(f"replay.load | source={directory}", "green")
                    self.load()
                    if hasattr(self.limiter, "set_avail_max"):
                        self.limiter.set_avail_max()

        self.batch_counts_sampled = []
        self.batch_priorities = []

    def set_train_ratio(self, train_ratio):
        sample_per_insert = train_ratio / self.batch_length
        self.limiter.set_samples_per_insert(sample_per_insert)

    def __len__(self):
        if self.finetune_online:
            return len(self.online_queue)
        else:
            return len(self.table) + len(self.table_pos)

    @property
    def is_train(self):
        return not self.is_eval

    @property
    def pos_stream_rate(self):
        if self.upsample_pos:
            return len(self.table_pos) / (len(self.table) + len(self.table_pos) + EPS)
        else:
            sum_pos_stream = sum(
                [v for k, v in self.counts_stream_reward.items() if k > 0]
            )
            if sum_pos_stream == 0:
                return 0
            sum_all = sum(self.counts_stream_reward.values())
            return sum_pos_stream / (sum_all + EPS)

    def eps_rate(self, val):
        sum_all = sum(self.counts_eps_reward.values())
        return max(0, self.counts_eps_reward[val]) / (sum_all + EPS)

    def reward_stream_rate(self, val):
        return self.counts_stream_reward[val] / (len(self.table) + EPS)

    def reward_balanced_weight(self, val):
        if self.config.replay.bw_scale == "none":
            return 1

        sum_eps = sum(self.counts_eps_reward.values())
        scale = sum_eps / (self.counts_eps_reward[val] + EPS)
        if self.config.replay.bw_scale == "sqrt":
            scale = np.sqrt(scale)
        elif self.config.replay.bw_scale == "cube_root":
            scale = np.cbrt(scale)
        elif self.config.replay.bw_scale != "none":
            raise ValueError(f"Unknown bw_scale {self.config.replay.bw_scale}")

        scale = max(scale, 1)
        return min(scale, self.config.replay.max_bw)

    @property
    def imbalanced_rewards(self):
        return self.counts_stream_reward.keys()

    @property
    def stats(self):
        def ratio(x, y):
            return x / y if y else np.nan

        m = self.metrics
        stats = {
            "size": len(self),
            "ram_gb": len(self) * self.itemsize / (1024**3),
            "materialized_ram_gb": self._materialized_nbytes / (1024**3),
            "inserts": m["inserts"],
            "samples": m["samples"],
            "insert_wait_avg": ratio(m["insert_wait_dur"], m["inserts"]),
            "insert_wait_frac": ratio(m["insert_wait_count"], m["inserts"]),
            "sample_wait_avg": ratio(m["sample_wait_dur"], m["samples"]),
            "sample_wait_frac": ratio(m["sample_wait_count"], m["samples"]),
            "pos_eps_rate": self.pos_eps_rate,  # how many positive rewards overall
            "pos_stream_rate": self.pos_stream_rate,
        }
        for k, v in self.counts_stream_reward.items():
            stats[f"{k}_stream_rate"] = self.reward_stream_rate(k)

        for k, v in self.counts_eps_reward.items():
            stats[f"{k}_eps_rate"] = self.eps_rate(k)
            stats[f"{k}_balanced_weight"] = self.reward_balanced_weight(k)
            # use

        if hasattr(self.limiter, "avail"):
            # assert isinstance(self.limiter, SamplesPerInsert), self.limiter
            stats.update(
                {
                    "avail": self.limiter.avail,
                    "avail_ratio": self.limiter.avail_ratio,
                    "max_avail": self.limiter.max_avail,
                }
            )
        for key in self.metrics:
            self.metrics[key] = 0
        return stats

    @staticmethod
    def preprocess(step, keep_keys=None):
        if keep_keys is not None:
            cprint(f"replay.keep_log_keys | keys={keep_keys}", "red")
        return {
            k: v
            for k, v in step.items()
            if not k.startswith("log_") or (keep_keys and k in keep_keys)
        }

    def _materialize_sequence(self, seq):
        materialized = {
            key: convert([step[key] for step in seq])
            for key in seq[0]
            if key not in self.dataset_exclude_keys and key != REPLAY_STEP_ID_KEY
        }

        for key in self._dataset_zero_keys:
            materialized[key] = np.zeros_like(materialized[key])

        if self.is_first:
            assert materialized["is_first"][0], materialized["is_first"]
        elif "is_first" in materialized:
            materialized["is_first"][0] = True

        if "reward_indicator" in materialized:
            assert min(materialized["reward_indicator"]) == max(
                materialized["reward_indicator"]
            ), (
                f'max={max(materialized["reward_indicator"])}, '
                f'min={min(materialized["reward_indicator"])}'
            )
            materialized["reward_indicator"] = materialized["reward_indicator"][0]

            if self.config.replay.imbalance_reward == "sum" and self.task == "s2":
                assert materialized["reward_indicator"] in [-1, 1.5, -0.5], (
                    materialized["reward_indicator"]
                )

        # Cached arrays are shared by samples. Batcher only reads and stacks
        # them; marking them read-only prevents a caller from corrupting replay.
        for value in materialized.values():
            if isinstance(value, np.ndarray):
                value.setflags(write=False)
        return materialized

    def _cache_materialized(self, key, materialized):
        self._materialized[key] = materialized
        self._materialized_nbytes += sum(
            value.nbytes
            for value in materialized.values()
            if isinstance(value, np.ndarray)
        )

    def _drop_materialized(self, key):
        materialized = self._materialized.pop(key, None)
        if materialized is not None:
            self._materialized_nbytes -= sum(
                value.nbytes
                for value in materialized.values()
                if isinstance(value, np.ndarray)
            )

    def add(
        self,
        step,
        worker=0,
        load=False,
        training=True,
        aug=False,  # doesn't count limiter insert for this
        reward_indicator=-1,
        is_read_step=False,  # doesn't count limiter insert for this
    ):
        # Persisted steps were already filtered before Saver.add(). Avoid copying
        # and filtering every step again while rebuilding replay from disk.
        if not load:
            step = self.preprocess(step, KEEP_KEYS)
        reward_indicator = (
            np.float32(reward_indicator)
            if load and self.is_first
            else np.round(reward_indicator, 2).astype(np.float32)
        )

        if step["is_first"]:
            # assert step['entity_pos'] (3, 3) is not all 0
            assert not np.all(step["entity_pos"] == 0), step["entity_pos"]

        # step["id"] = np.asarray(uuid(step.get("id")))
        if "id" in step:
            step.pop("id")

        stream = self.streams[reward_indicator][worker]
        if not self.finetune_policy:
            if step["is_first"]:
                # try:
                if not (len(stream) == 0 or stream[-1]["is_last"] == True):
                    self.error_cnt_is_first += 1
                    cprint(
                        f"replay.invalid_episode_start | "
                        f"count={self.error_cnt_is_first}",
                        "red",
                    )
                # except AssertionError as e:
            else:
                if len(stream) > 0:
                    assert stream[-1]["is_last"] == False, stream[-1]

        step["reward_indicator"] = reward_indicator

        # Saver returns provenance derived from chunk UUID and offset. Loader
        # injects the same value without storing it as a model input field.
        step_id = step.pop(REPLAY_STEP_ID_KEY, None)
        if not load and self.is_train and self.saver:
            step_id = self.saver.add(step, worker)
        if step_id is None:
            step_id = np.frombuffer(uuid().value, np.uint8).copy()
        step[REPLAY_STEP_ID_KEY] = np.asarray(step_id, np.uint8)

        stream.append(step)
        # not count when augumenting an eps
        self.counts_insert_step[reward_indicator][worker] += (
            aug == False and is_read_step == False
        )  # not count when augumenting an eps

        assert self.config.replay.imbalance_reward in [
            "sum",
            "last",
        ], self.config.replay.imbalance_reward
        # if self.config.replay.imbalance_reward == "sum":
        # if self.task == "s2":
        #     assert reward_indicator in [
        #         -1,
        #         1.5,
        #         -0.5], reward_indicator

        self.counters[reward_indicator][worker] += 1
        if self.finetune_online:
            self.online_counters[reward_indicator][worker] += 1
            if len(stream) >= self.batch_length and (
                self.online_counters[reward_indicator][worker] >= self.online_stride
            ):
                self.online_queue.append(tuple(stream))
                self.online_counters[reward_indicator][worker] = 0

        # continue only IF
        # len(stream) >= self.length = batch_length
        # AND self.counters[worker] >= self.stride = 1
        if (
            len(stream) < self.batch_length
            or self.counters[reward_indicator][worker] < self.stride
        ):
            return

        # take first element of dequeue stream, make sure is_first=True
        if self.is_first:
            if not stream[0]["is_first"]:
                while len(stream) > 0 and not stream[0]["is_first"]:
                    stream.popleft()
                    # self.count_popleft[worker] += 1
                return

        # if self.overfit_eps:
        #     if self.capacity and len(self) >= self.capacity:
        #         cprint(f"Replay is full: {len(self)} >= {self.capacity}", "red")
        #         return

        # ADD this stream to global buffer
        self.counters[reward_indicator][worker] = 0
        seq = tuple(stream)  # tuple of dict
        key = uuid(np.asarray(seq[0][REPLAY_STEP_ID_KEY], np.uint8))

        from ledwm.embodied.replay.limiters import SamplesPerInsert
        from ledwm.embodied.replay.limiters import MinSize

        if load:
            assert self.limiter.want_load()[0]
        else:
            if isinstance(self.limiter, MinSize):
                dur = wait(self.limiter.want_insert, "replay.wait_insert")
            else:
                assert isinstance(self.limiter, SamplesPerInsert)
                dur = wait(
                    bind(
                        self.limiter.want_insert,
                        x=self.counts_insert_step[reward_indicator][worker],
                    ),
                    "replay.wait_insert",
                )
            self.metrics["inserts"] += 1
            self.metrics["insert_wait_dur"] += dur
            self.metrics["insert_wait_count"] += int(dur > 0)

        if self.is_first:
            assert seq[0]["is_first"], seq[0]["is_first"]

        materialized = None
        if not self.finetune_online:
            materialized = self._materialize_sequence(seq)

        with self.lock:
            if reward_indicator > 0 and self.upsample_pos:
                self.table_pos[key] = seq
                self.remover_pos[key] = seq
                self.sampler_pos[key] = seq
            else:
                self.table[key] = seq
                self.remover[key] = seq
                self.sampler[key] = seq
            if materialized is not None:
                self._cache_materialized(key, materialized)

            # print(f"DIFF add: {set(self.table.keys()) - set(self.remover.queue)}")
            self.counts_insert_step[reward_indicator][worker] = 0

            if not self.is_eval:
                self.counts_stream_reward[reward_indicator] += 1
                self.counts_eps_reward[reward_indicator] += sum(
                    bool(item["is_last"]) for item in seq
                )

        if self.is_train:
            self._update_fill_progress(load)

        # maintain added eps in the stream
        # after add the stream, delete until unfinished eps
        # if last_step["is_last"] == False:
        # delete until the last step["is_first"] == True
        # else:
        assert len(stream) > 0, "stream is empty"
        if stream[-1]["is_last"] or self.finetune_policy:
            # clear the stream -> this stream (for this worker and reward_indicator) is done
            stream.clear()
        else:
            new_stream = deque(maxlen=self.batch_length)
            while len(stream) > 0 and stream[-1]["is_last"] == False:
                new_stream.appendleft(stream.pop())

            assert new_stream[0]["is_first"], new_stream[0]
            self.streams[reward_indicator][worker] = new_stream
            # else:

        while self.is_full:
            if not self.is_eval:
                event_logger.debug(
                    f"replay.evict | reason=over_capacity | "
                    f"buffer_size={len(self)} | capacity={self.size} | "
                    f"reward={reward_indicator}"
                )
            self._remove()

    def _update_fill_progress(self, load):
        """Show replay readiness once instead of logging every N sequences."""
        total = int(self.min_size)
        current = min(len(self), total)

        # Disk restore has its own chunk progress bar. Remember when restored
        # data already makes the replay ready so live inserts do not open a
        # redundant 100% bar.
        if load:
            if current >= total:
                self._fill_progress_done = True
            return

        with self._fill_progress_lock:
            if self._fill_progress_done:
                return
            if self._fill_progress is None:
                self._fill_progress = tqdm(
                    total=total,
                    initial=current,
                    desc="replay.fill",
                    unit="seq",
                    dynamic_ncols=True,
                    mininterval=0.25,
                )
            else:
                delta = current - self._fill_progress.n
                if delta > 0:
                    self._fill_progress.update(delta)

            if current >= total:
                self._fill_progress.close()
                self._fill_progress = None
                self._fill_progress_done = True

    def print(self, event="add_seq", color=None):
        mode = "train" if self.is_train else "eval"
        reward_counts = {
            float(reward): int(count)
            for reward, count in sorted(
                self.counts_eps_reward.items(), key=lambda item: float(item[0])
            )
        }
        cprint(
            f"replay.{event} | mode={mode} | buffer_size={len(self)} | "
            f"pos_eps_rate={self.pos_eps_rate:.3f} | "
            f"pos_stream_rate={self.pos_stream_rate:.3f} | "
            f"reward_counts={reward_counts}",
            color,
        )

    @property
    def is_full(self):
        return self.size and len(self) > self.size

    @property
    def pos_eps_rate(self):
        sum_pos_reward = sum([v for k, v in self.counts_eps_reward.items() if k > 0])
        if sum_pos_reward == 0:
            return 0
        return sum_pos_reward / sum(self.counts_eps_reward.values())

    def sample(self):
        dur = wait(
            self.limiter.want_sample,
            f"replay.wait_sample | mode={'train' if self.is_train else 'eval'}",
        )
        self.metrics["samples"] += 1
        self.metrics["sample_wait_dur"] += dur
        self.metrics["sample_wait_count"] += int(dur > 0)

        seq = None
        sample_sampler = None
        if self.finetune_online:
            # try:
            seq = self.online_queue.popleft()
        else:
            not_sampled = True
            if self.upsample_pos:
                do_upsample = np.random.rand() < self.upsample_pos_rate
            else:
                do_upsample = False

            with self.lock:
                while not_sampled:
                    try:
                        if do_upsample and len(self.table_pos) > 0:
                            sample_sampler = self.sampler_pos
                            sample_seq_id = sample_sampler.__call__()
                            seq = self._materialized[sample_seq_id]

                        elif len(self.table) > 0:
                            sample_sampler = self.sampler
                            sample_seq_id = sample_sampler.__call__()
                            seq = self._materialized[sample_seq_id]

                        else:
                            sample_sampler = self.sampler_pos
                            sample_seq_id = sample_sampler.__call__()
                            seq = self._materialized[sample_seq_id]

                        not_sampled = False

                    except KeyError:
                        cprint(
                            f"replay.sample_retry | reason=missing_key | "
                            f"upsample_positive={do_upsample}",
                            "red",
                        )

        if not self.finetune_online:
            with self.lock:
                self.counts_stream2sample[sample_seq_id] += 1
                sample_count = self.counts_stream2sample[sample_seq_id]
                if self.is_prioritize_replay:
                    assert sample_sampler is not None
                    sample_priority = sample_sampler.key2priority[sample_seq_id]
                else:
                    sample_priority = None

        assert seq is not None, "No sequence found."
        if self.finetune_online:
            seq = self._materialize_sequence(seq)
        else:
            # Return a fresh mapping so adding sample metadata cannot mutate the
            # cache. Array values remain shared and read-only until np.stack().
            seq = dict(seq)

        if not self.finetune_online:
            seq["sample_id"] = sample_seq_id
            seq[REPLAY_SAMPLE_COUNT_KEY] = np.asarray(sample_count)
            if sample_priority is not None:
                seq[REPLAY_SAMPLE_PRIORITY_KEY] = np.asarray(sample_priority)

        # last_reward = None
        # if self.config.replay.seperate_reward_stream:
        #     last_rewards = [
        #         reward for i, reward in enumerate(seq["reward"]) if seq["is_last"][i]
        #     ]
        #     last_reward = last_rewards[0]
        #     assert max(last_rewards) == min(last_rewards) == last_reward, last_rewards

        # if self.config.replay.imbalance == "balanced_weight" and not self.is_eval:
        # if self.config.replay.imbalance == "balanced_weight":

        if (
            not self.finetune_online
            and self.is_train
            and self.counts_stream2sample[sample_seq_id] >= self.train_ratio
        ):
            if not self.is_eval:  # only train
                event_logger.debug(
                    f"replay.evict | reason=sample_limit | key={sample_seq_id} | "
                    f"sample_count={self.counts_stream2sample[sample_seq_id]} | "
                    f"train_ratio={self.train_ratio}"
                )
            self._remove_key(
                sample_seq_id,
                seq["reward_indicator"],
                force=True,
                preserve_sample_count=True,
            )

        return seq

    def sample_batch(self, batch_size):
        """Sample one prioritized learner batch without replacement."""
        batch_size = int(batch_size)
        assert batch_size > 0, batch_size
        assert self.supports_unique_batch_sampling

        def enough_unique_sequences():
            with self.lock:
                available = len(self.sampler)
            return (
                available >= batch_size,
                f"need unique batch: {available} < {batch_size}",
            )

        wait(
            enough_unique_sequences,
            f"replay.wait_sample | mode={'train' if self.is_train else 'eval'}",
        )

        wait_dur = 0.0
        wait_count = 0
        for _ in range(batch_size):
            dur = wait(
                self.limiter.want_sample,
                f"replay.wait_sample | mode={'train' if self.is_train else 'eval'}",
            )
            wait_dur += dur
            wait_count += int(dur > 0)
        self.metrics["samples"] += batch_size
        self.metrics["sample_wait_dur"] += wait_dur
        self.metrics["sample_wait_count"] += wait_count

        with self.lock:
            sample_ids = self.sampler.sample_batch(batch_size)
            assert len(sample_ids) == len(set(sample_ids)), sample_ids
            samples = []
            for sample_id in sample_ids:
                seq = dict(self._materialized[sample_id])
                self.counts_stream2sample[sample_id] += 1
                sample_count = self.counts_stream2sample[sample_id]
                sample_priority = self.sampler.key2priority[sample_id]
                seq["sample_id"] = sample_id
                seq[REPLAY_SAMPLE_COUNT_KEY] = np.asarray(sample_count)
                seq[REPLAY_SAMPLE_PRIORITY_KEY] = np.asarray(sample_priority)
                samples.append(seq)
            exhausted = [
                sample_id
                for sample_id in sample_ids
                if self.counts_stream2sample[sample_id] >= self.train_ratio
            ]
        for sample_id, seq in zip(sample_ids, samples):
            if sample_id not in exhausted:
                continue
            self._remove_key(
                sample_id,
                seq.get("reward_indicator"),
                force=True,
                preserve_sample_count=True,
            )
        return samples

    def __getitem__(self, idx):
        sample_seq_id = list(self.table.keys())[idx]
        return dict(self._materialized[sample_seq_id])

    @property
    def num_samples_whole_dataset(self):
        """
        return number of samples based on ordered dataset
        """
        return [self.counts_stream2sample[k] for k in list(self.table.keys())]

    @property
    def priority_dataset(self):
        if self.is_prioritize_replay:
            with self.lock:
                # TODO still mutuate during training?
                keys = list(self.table.keys())  # Create a static list of keys
            from ledwm.embodied.replay.Prioritized import PrioritizedSampler

            assert isinstance(self.sampler, PrioritizedSampler), type(self.sampler)
            with self.lock:
                return [self.sampler.key2priority[k] for k in keys]
        else:
            return None

    @property
    def priority_loss_dataset(self):
        if self.is_prioritize_replay:
            keys = list(self.table.keys())
            from ledwm.embodied.replay.Prioritized import PrioritizedSampler

            assert isinstance(self.sampler, PrioritizedSampler), type(self.sampler)
            return [self.sampler.get_model_priority(k) for k in keys]
        else:
            return None

    @property
    def is_prioritize_replay(self):
        from ledwm.embodied.replay.Prioritized import PrioritizedSampler

        return isinstance(self.sampler, PrioritizedSampler)

    @property
    def supports_unique_batch_sampling(self):
        return (
            self.is_prioritize_replay
            and not self.finetune_online
            and not self.upsample_pos
            and not self.config.replay.remove_oversample
        )

    def clear_after_batch(self):
        with self.lock:
            self.batch_counts_sampled.clear()
            self.batch_priorities.clear()

    def _remove_key(
        self,
        key,
        reward_indicator=None,
        force=False,
        preserve_sample_count=False,
    ):
        with self.lock:
            if key not in self.table and (
                not self.upsample_pos or key not in self.table_pos
            ):
                event_logger.debug(
                    f"replay.evict_skipped | reason=already_removed | key={key}"
                )
                return

            # Keep limiter accounting and replay removal atomic. Otherwise a
            # capacity eviction can remove this key after sampling releases the
            # replay lock but before its deferred sample-limit eviction runs.
            wait(self.limiter.want_remove, "Replay remove is waiting")
            if self.upsample_pos:
                if key in self.table_pos and (
                    force or len(self.table_pos) >= self.min_size
                ):
                    if reward_indicator is not None and not self.is_eval:
                        reward_indicator = reward_indicator.astype(np.float32)
                        assert reward_indicator in self.counts_stream_reward, (
                            reward_indicator
                        )
                        assert reward_indicator in self.counts_eps_reward, (
                            reward_indicator
                        )
                        self.counts_stream_reward[reward_indicator] -= 1
                        self.counts_eps_reward[reward_indicator] -= self.count_num_eps(
                            self.table_pos, key
                        )
                        self.counts_stream_reward[reward_indicator] = max(
                            0, self.counts_stream_reward[reward_indicator]
                        )
                        self.counts_eps_reward[reward_indicator] = max(
                            0, self.counts_eps_reward[reward_indicator]
                        )
                        assert self.counts_stream_reward[reward_indicator] >= 0, (
                            self.counts_stream_reward[reward_indicator]
                        )
                        assert self.counts_eps_reward[reward_indicator] >= 0, (
                            self.counts_eps_reward[reward_indicator]
                        )

                        # assert (
                        #     len(self.counts_eps_reward) <= 2
                        # ), f"{self.counts_eps_reward=}"
                        # assert (
                        #     len(self.counts_stream_reward) <= 2
                        # ), f"{self.counts_stream_reward=}"

                    if key in self.table_pos:
                        del self.table_pos[key]
                    self._drop_materialized(key)
                    if key in self.remover_pos:
                        del self.remover_pos[key]
                    if key in self.sampler_pos:
                        del self.sampler_pos[key]
                    if not preserve_sample_count:
                        self.counts_stream2sample[key] = 0
                    return
            if not force and len(self.table) < self.min_size:
                event_logger.debug(
                    f"replay.evict_skipped | reason=below_min_size | "
                    f"table_size={len(self.table)} | min_size={self.min_size}"
                )
                return

            if not preserve_sample_count:
                self.counts_stream2sample[key] = 0
            if reward_indicator is not None and self.is_train:
                reward_indicator = reward_indicator.astype(np.float32)
                assert reward_indicator in self.counts_stream_reward, reward_indicator
                assert reward_indicator in self.counts_eps_reward, reward_indicator
                self.counts_stream_reward[reward_indicator] -= 1
                self.counts_stream_reward[reward_indicator] = max(
                    0, self.counts_stream_reward[reward_indicator]
                )
                self.counts_eps_reward[reward_indicator] -= self.count_num_eps(
                    self.table, key
                )
                self.counts_eps_reward[reward_indicator] = max(
                    0, self.counts_eps_reward[reward_indicator]
                )
            if key in self.table:
                del self.table[key]
            self._drop_materialized(key)
            if key in self.remover:
                del self.remover[key]
            if key in self.sampler:
                del self.sampler[key]

    def save_progress(self):
        """Return checkpointable sample counts and exact limiter debt."""
        with self.lock:
            sample_counts = {
                str(key): int(count)
                for key, count in self.counts_stream2sample.items()
                if count > 0
            }
        limiter_avail = None
        if hasattr(self.limiter, "avail"):
            with self.limiter.lock:
                limiter_avail = self.limiter.avail
        return {
            "version": 1,
            "sample_counts": sample_counts,
            "limiter_avail": limiter_avail,
        }

    def load_progress(self, state):
        """Restore remaining lifetime sample budgets after raw replay load."""
        if not state:
            cprint("replay.progress_restore | state=missing", "yellow")
            return
        if int(state.get("version", 0)) != 1:
            raise ValueError(f"Unsupported replay progress version: {state}")

        saved_counts = state.get("sample_counts", {})
        with self.lock:
            active_keys = list(self.table) + list(self.table_pos)
            restored = 0
            for key in active_keys:
                count = int(saved_counts.get(str(key), 0))
                self.counts_stream2sample[key] = count
                sampler = self.sampler_pos if key in self.table_pos else self.sampler
                if hasattr(sampler, "key2visit_count"):
                    sampler.key2visit_count[key] = count
                restored += count > 0

        exhausted = [
            key
            for key in active_keys
            if self.counts_stream2sample[key] >= self.train_ratio
        ]
        for key in exhausted:
            table = self.table_pos if key in self.table_pos else self.table
            reward_indicator = None
            if key in table:
                reward_indicator = table[key][0].get("reward_indicator")
            self._remove_key(
                key,
                reward_indicator,
                force=True,
                preserve_sample_count=True,
            )

        limiter_avail = state.get("limiter_avail")
        if limiter_avail is not None and hasattr(self.limiter, "avail"):
            with self.limiter.lock:
                self.limiter.avail = float(limiter_avail)
        cprint(
            f"replay.progress_restore | restored={restored} | "
            f"exhausted={len(exhausted)} | limiter_avail={limiter_avail}",
            "green",
        )

    def _remove(self):
        wait(self.limiter.want_remove, "replay.wait_remove")
        with self.lock:
            if self.upsample_pos:
                if np.random.rand() < self.pos_stream_rate:
                    key = self.remover_pos.__call__()
                    reward_indicator = (
                        self.table_pos[key][0]["reward_indicator"]
                        if "reward_indicator" in self.table_pos[key][0]
                        else None
                    )
                    if reward_indicator is not None and self.is_train:
                        # last_reward = np.round(last_reward, 2)
                        assert reward_indicator in self.counts_stream_reward, (
                            reward_indicator
                        )
                        assert reward_indicator in self.counts_eps_reward, (
                            reward_indicator
                        )
                        self.counts_stream_reward[reward_indicator] -= 1
                        self.counts_stream_reward[reward_indicator] = max(
                            0, self.counts_stream_reward[reward_indicator]
                        )

                        self.counts_eps_reward[reward_indicator] -= self.count_num_eps(
                            self.table_pos, key
                        )
                        self.counts_eps_reward[reward_indicator] = max(
                            0, self.counts_eps_reward[reward_indicator]
                        )

                    del self.remover_pos[key]
                    if key in self.table_pos:
                        del self.table_pos[key]
                    self._drop_materialized(key)
                    if key in self.sampler_pos:
                        del self.sampler_pos[key]
                    self.counts_stream2sample[key] = 0
                    return

            key = self.remover.__call__()
            reward_indicator = (
                self.table[key][0]["reward_indicator"]
                if "reward_indicator" in self.table[key][0]
                else None
            )
            if reward_indicator is not None and self.is_train:
                assert reward_indicator in self.counts_stream_reward, reward_indicator
                assert reward_indicator in self.counts_eps_reward, reward_indicator
                self.counts_stream_reward[reward_indicator] -= 1
                self.counts_stream_reward[reward_indicator] = max(
                    0, self.counts_stream_reward[reward_indicator]
                )
                # if key in self.table:
                self.counts_eps_reward[reward_indicator] -= self.count_num_eps(
                    self.table, key
                )
                self.counts_eps_reward[reward_indicator] = max(
                    0, self.counts_eps_reward[reward_indicator]
                )

            del self.remover[key]
            del self.table[key]
            self._drop_materialized(key)
            del self.sampler[key]
            self.counts_stream2sample[key] = 0

    def count_num_eps(self, table, key):
        seq = table[key]
        return sum(e["is_last"] for e in seq)

    def dataset(self):
        while True:
            yield self.sample()

    def prioritize(self, ids, prios):
        if hasattr(self.sampler, "prioritize"):
            self.sampler.update_priority(ids, prios)

    def save(self, wait=False):
        assert not self.is_eval, "Cannot save eval replay buffer."
        if not self.saver:
            return
        self.saver.save(wait)

    def load(self, data=None):
        assert self.is_train, "Cannot load eval replay buffer."
        if not self.saver:
            return

        if "offline_wm" in self.config.run.script:
            self.verbose_every = 500

        workers = set()
        is_debug = self.config.run.debug or self.config.overfit_batch
        for step, worker in self._load_complete_episodes(
            self.saver, self.size, self.batch_length, is_debug
        ):
            workers.add(worker)
            # if step["reward_indicator"] != -2:
            self.add(
                step,
                worker,
                load=True,
                reward_indicator=(step["reward_indicator"]),
            )

        self.print(event="load_done", color="green")

        if "offline_wm" in self.config.run.script:
            assert self.__len__() > 0, (
                f"Replay buffer is empty at {self.saver.directory}."
            )

        self._clear_loaded_workers(workers)

    def _load_complete_episodes(self, load_saver, capacity, batch_length, debug=False):
        """Repack complete labeled episodes so sparse reward streams survive reload."""
        if not self.is_first:
            yield from load_saver.load(capacity, batch_length, debug)
            return

        pending = defaultdict(list)
        packed_workers = {}
        loaded_episodes = defaultdict(int)
        dropped_fragments = 0
        for step, source_worker in load_saver.load(capacity, batch_length, debug):
            if "reward_indicator" not in step:
                yield step, source_worker
                continue

            is_first = bool(step["is_first"])
            is_last = bool(step["is_last"])
            if is_first:
                if pending[source_worker]:
                    dropped_fragments += 1
                pending[source_worker] = []
            elif not pending[source_worker]:
                dropped_fragments += 1
                continue

            pending[source_worker].append(step)
            if not is_last:
                continue

            episode = pending.pop(source_worker)
            indicators = {
                float(_canonical_reward_indicator(item["reward_indicator"]))
                for item in episode
            }
            if len(indicators) != 1:
                raise ValueError(
                    f"Replay episode has mixed reward indicators: {indicators}"
                )
            reward_indicator = indicators.pop()
            for item in episode:
                item["reward_indicator"] = np.float32(reward_indicator)
            packed_worker = packed_workers.setdefault(
                reward_indicator, ("replay_load", reward_indicator)
            )
            loaded_episodes[reward_indicator] += 1
            for item in episode:
                yield item, packed_worker

        dropped_fragments += sum(bool(episode) for episode in pending.values())
        print(
            f"replay.load_repacked | episodes={dict(loaded_episodes)} | "
            f"dropped_fragments={dropped_fragments}"
        )

    def _clear_loaded_workers(self, workers):
        for mapping in (self.streams, self.counters, self.counts_insert_step):
            for worker_map in mapping.values():
                for worker in workers:
                    worker_map.pop(worker, None)

    def preload_from_dir(self, load_dir):
        """Preload the replay with episodes from another `load_dir`."""
        from . import saver

        load_saver = saver.Saver(
            load_dir, self.chunks, self.size - len(self), self.batch_length
        )
        assert len(self) < self.size, "Replay is already full."
        print(
            f"replay.preload | source={load_saver.directory} | "
            f"buffer_size={len(self):.2E} | capacity={self.size:.2E}"
        )
        workers = set()
        steps = 0
        for step, worker in self._load_complete_episodes(
            load_saver, self.size - len(self), self.batch_length
        ):
            workers.add(worker)
            self.add(
                step,
                worker,
                load=True,
                reward_indicator=step.get("reward_indicator", -1),
            )
            steps += 1

        self._clear_loaded_workers(workers)
        del load_saver

        print(f"replay.preload_done | source={load_dir} | steps={steps}")
        self.preloaded[str(load_dir)] = steps

    def reset(self):
        # empty replay
        with self._fill_progress_lock:
            if self._fill_progress is not None:
                self._fill_progress.close()
            self._fill_progress = None
            self._fill_progress_done = False
        self.table.clear()
        if hasattr(self, "table_pos"):
            self.table_pos.clear()
        self._materialized.clear()
        self._materialized_nbytes = 0
        self.streams.clear()
        self.counts_insert_step.clear()
        self.counters.clear()
        self.counts_stream2sample.clear()
        self.counts_eps_reward.clear()
        self.counts_eps.clear()
        self.counts_stream_reward.clear()
        self.batch_counts_sampled.clear()
        self.batch_priorities.clear()
