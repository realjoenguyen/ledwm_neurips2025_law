import concurrent.futures
import os
import time
from typing import TYPE_CHECKING, Callable
from termcolor import cprint
from ledwm.WM import REWARD_VALUES
import ledwm.Optimizer
try:
    from ledwm.startup import configure_tensorflow_cpp_warnings
except ModuleNotFoundError:
    from startup import configure_tensorflow_cpp_warnings

from torch.utils.data import DataLoader

if TYPE_CHECKING:
    from ledwm.agent import Agent
    from ledwm.embodied.run.parallel import Dataset

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from . import jaxutils
from . import ninjax as nj
from ledwm.embodied.core.base import Agent as BaseAgent
from ledwm.embodied.core.path import Path
from ledwm.embodied.core.counter import Counter
from ledwm.embodied.core.when import Every
from ledwm.embodied.replay.generic import (
    REPLAY_SAMPLE_COUNT_KEY,
    REPLAY_SAMPLE_PRIORITY_KEY,
)
from ledwm.embodied.run.timing import has_tree_leaves, timing_enabled

tree_map = jax.tree_util.tree_map
tree_flatten = jax.tree_util.tree_flatten
tree_leaves = jax.tree_util.tree_leaves


def _stack_for_sharding(xs):
    stack = jnp.stack if any(isinstance(x, jax.Array) for x in xs) else np.stack
    return stack(xs)


def _policy_state_leaf_to_batch(x):
    if isinstance(x, list):
        if not x:
            return np.asarray(x)
        if any(isinstance(item, jax.Array) for item in x):
            return jnp.stack(x)
        return np.asarray(x)
    if isinstance(x, jax.Array):
        return x
    return np.asarray(x)


def _device_put_sharded(shards, devices):
    put_sharded = getattr(jax, "device_put_sharded", None)
    if put_sharded is not None:
        return put_sharded(shards, devices)
    mesh = Mesh(np.array(devices), ("i",))
    sharding = NamedSharding(mesh, P("i"))
    return tree_map(lambda *xs: jax.device_put(_stack_for_sharding(xs), sharding), *shards)


def _device_put_replicated(value, devices):
    put_replicated = getattr(jax, "device_put_replicated", None)
    if put_replicated is not None:
        return put_replicated(value, devices)
    mesh = Mesh(np.array(devices), ("i",))
    sharding = NamedSharding(mesh, P("i"))
    return tree_map(
        lambda x: jax.device_put(
            _stack_for_sharding([x] * len(devices)), sharding
        ),
        value,
    )


def _unreplicate_for_policy(x):
    x = x[0]
    addressable_data = getattr(x, "addressable_data", None)
    if callable(addressable_data):
        return addressable_data(0)
    device_buffer = getattr(x, "device_buffer", None)
    if device_buffer is not None:
        return device_buffer
    return x


def skip_train_outs_enabled():
    return os.environ.get("LEDWM_SKIP_TRAIN_OUTS") == "1"


def Wrapper(agent_cls):
    class Agent(JAXAgent):
        configs = agent_cls.configs
        inner = agent_cls

        def __init__(self, *args, **kwargs):
            super().__init__(agent_cls, *args, **kwargs)

    return Agent


class JAXAgent(BaseAgent):
    def __init__(
        self,
        agent_cls,
        obs_space,
        act_space,
        step,
        config,
        env_cache=None,
        reward_values=None,
    ):
        self.config_jax = config.jax
        self.priority = config.replay.type == "prioritize"
        if self.priority:
            cprint("agent.replay | type=prioritized", "yellow")
        self.batch_size = config.batch_size
        self.batch_length = config.batch_length
        self.data_workers = config.data_workers
        self.logdir = Path(config.logdir)
        self._setup()

        available = jax.devices(self.config_jax.platform)
        print(
            f"agent.devices_available | platform={self.config_jax.platform} | "
            f"count={len(available)} | devices={available}"
        )
        self.policy_devices = [available[i] for i in self.config_jax.policy_devices]
        self.train_devices = [available[i] for i in self.config_jax.train_devices]
        self.single_device = (self.policy_devices == self.train_devices) and (
            len(self.policy_devices) == 1
        )
        print(
            f"agent.devices | local_count={jax.local_device_count()} | "
            f"available={available}"
        )
        cprint(f"agent.devices | role=policy | devices={self.policy_devices}")
        cprint(f"agent.devices | role=train | devices={self.train_devices}")

        if config.reward_head.dist == "onehot":
            with jax.transfer_guard("allow"):
                reward_values = jnp.array(REWARD_VALUES[config.task])

        self.agent: "Agent" = agent_cls(
            obs_space, act_space, step, config, env_cache, reward_values, name="agent"
        )
        self.rng = np.random.default_rng(config.seed)
        self._obs_spaces = obs_space
        self._train_spaces = {**obs_space, **act_space}

        self._transform()
        self.varibs = self._init_varibs(obs_space, act_space)  # model weights
        self.updates = Counter()
        self.once = True

        self.outs_worker = concurrent.futures.ThreadPoolExecutor(1)
        self.mets_worker = concurrent.futures.ThreadPoolExecutor(1)
        self.sync_worker = concurrent.futures.ThreadPoolExecutor(1)
        self.outs_promise = None
        self.mets_promise = None
        self.sync_promise = None
        self.last_policy_timing = None

        self.should_sync = Every(self.config_jax.sync_every)
        if not self.single_device:
            print("agent.variables_sync | destination=policy_devices")
            self.policy_varibs = self._copy_varibs(self.varibs)

        # self.load_wm_only = getattr(config, "load_wm_only", False)
        # if self.load_wm_only:
        #     print("Only loading WM from agent ckpt.")
        self.is_loading_partial = config.load_only_key != ""
        self.load_match_shape = getattr(config, "load_match_shape", False)
        self.exclude_key = getattr(config, "load_exclude_key", "")
        self.config = config

    def policy(
        self,
        obs,  # num_envs, d*
        state=None,
        step=None,  # -1 if eval, test
        mode="train",
        return_state_to_host=True,
    ):
        # this is used in parallel_actor
        policy_timing = (
            {}
            if timing_enabled(getattr(self.config, "run", None))
            else None
        )

        def record(name, start):
            now = time.perf_counter()
            if policy_timing is not None:
                policy_timing[name] = policy_timing.get(name, 0.0) + (now - start)
            return now

        phase_start = time.perf_counter()
        obs = {k: v for k, v in obs.items() if not k.startswith("log_")}
        obs = self.dataload_preprocess_single(obs)
        phase_start = record("preprocess", phase_start)
        obs = self._convert_inps(obs, self.policy_devices)
        phase_start = record("obs_device_put", phase_start)
        rng = self._next_rngs(self.policy_devices)
        phase_start = record("rng", phase_start)
        varibs = self.varibs if self.single_device else self.policy_varibs

        if state is None:
            # state = agent.Agent.policy_initial: wm.initial, policy.initial, expl.initial
            # out, state for pure function. Get "out"
            state, _ = self._policy_initial(varibs, rng, obs["is_first"])
            phase_start = record("state_init", phase_start)

        else:
            state = tree_map(
                _policy_state_leaf_to_batch,
                state,
                is_leaf=lambda x: isinstance(x, list),
            )
            phase_start = record("state_preprocess", phase_start)
            state = self._convert_inps(state, self.policy_devices)
            phase_start = record("state_device_put", phase_start)

        # if "train" not in mode:
        # step = -1
        if step is not None:
            with jax.transfer_guard("allow"):
                # if step is int then
                if isinstance(step, int):
                    n_devices = len(self.policy_devices)
                    step = jnp.repeat(jnp.array(step, jnp.int32), n_devices).reshape(
                        n_devices
                    )
                step = self._convert_inps(step, self.policy_devices)
            phase_start = record("step_device_put", phase_start)

        (outs, state, info), _ = self._policy(
            varibs,
            rng,
            obs,
            state,
            step=step,
            mode=mode,
        )  # type: ignore # type: Agent.policy
        phase_start = record("jit_policy_call", phase_start)

        outs = jax.block_until_ready(outs)
        phase_start = record("block_until_ready", phase_start)

        if not self.single_device:
            if self.sync_promise and self.sync_promise.done():
                self.policy_varibs = self.sync_promise.result()
                self.sync_promise = None
        phase_start = record("sync_handoff", phase_start)

        outs = self._convert_outs(outs, self.policy_devices)
        phase_start = record("outs_device_get", phase_start)
        if return_state_to_host:
            state = self._convert_outs(state, self.policy_devices)
            phase_start = record("state_device_get", phase_start)
        else:
            phase_start = record("state_keep_device", phase_start)
        if has_tree_leaves(info):
            info = self._convert_outs(info, self.policy_devices)
            record("info_device_get", phase_start)
        self.last_policy_timing = policy_timing

        return outs, state, info

    def finetune_policy(
        self,
        data,
        state=None,
        step: int = None,  # type: ignore
        uncertainty=None,
    ):
        data = data.copy()
        rng = data.pop("rng")
        if state is None:
            rng = self._next_rngs(self.train_devices)
            state, self.varibs = self._train_initial(self.varibs, rng, data["is_first"])

        prev_varibs = self.varibs
        #  Agent.train
        if step is not None:
            with jax.transfer_guard("allow"):
                # n_devices = len(self.train_devices)
                bs = data[list(data.keys())[0]].shape[0]
                step = jnp.repeat(jnp.array(step, jnp.int32), bs).reshape(bs)
                step = self._convert_inps(step, self.train_devices)

        if uncertainty is not None:
            with jax.transfer_guard("allow"):
                n_devices = len(self.train_devices)
                uncertainty = jnp.repeat(
                    jnp.array(uncertainty, jnp.float32), n_devices
                ).reshape(n_devices)  # type: ignore
                uncertainty = self._convert_inps(uncertainty, self.train_devices)

        (outs, state, mets), self.varibs = self._finetune_policy(
            self.varibs,
            rng,
            data,
            state,
            step,
            uncertainty,
        )
        self.updates.increment()

        if not self.single_device:
            if not self.sync_promise and self.should_sync(self.updates):
                self.sync_promise = self.sync_worker.submit(
                    self._copy_varibs, prev_varibs, block=True
                )

        return_outs = {}
        if self.outs_promise:
            eturn_outs = self.outs_promise.result()
        self.outs_promise = self.outs_worker.submit(
            self._convert_outs, outs, self.train_devices
        )

        return_mets = {}
        if self.mets_promise and self.mets_promise.done():
            return_mets = self.mets_promise.result()
            self.mets_promise = None

        if not self.mets_promise:
            # Only request metrics if we aren't currently waiting for previous
            # metrics. This means we'll skip the metrics of some training steps if
            # fetching them from device would slow down the training loop.
            self.mets_promise = self.mets_worker.submit(
                self._convert_mets, mets, self.train_devices
            )

        if self.once:
            self.once = False
            assert ledwm.Optimizer.Optimizer.PARAM_COUNTS
            for name, count in ledwm.Optimizer.Optimizer.PARAM_COUNTS.items():
                if count is None:
                    cprint(f"agent.param_count_missing | name={name}", "red")
                    continue
                return_mets[f"params_{name}"] = float(count)

        if self.config_jax.profiler:
            outdir, copyto = self.logdir, None
            if str(outdir).startswith(("gs://", "/gcs/")):
                copyto = outdir
                outdir = Path("/tmp/profiler")
                outdir.mkdirs()

            if self.updates == 100:
                print(f"jax.profiler_start | directory={outdir}")
                jax.profiler.start_trace(str(outdir))

            if self.updates == 120:
                from ledwm.embodied.core import path as pathlib

                print("jax.profiler_stop")
                jax.profiler.stop_trace()
                if copyto:
                    pathlib.GFilePath(outdir).copy(copyto)
                    print(f"jax.profiler_copy | source={outdir} | destination={copyto}")

        return return_outs, state, return_mets

    def _process_train_outs(self, outs):
        if skip_train_outs_enabled():
            self.outs_promise = None
            return {}

        return_outs = {}
        if self.outs_promise:
            return_outs = self.outs_promise.result()
        self.outs_promise = self.outs_worker.submit(
            self._convert_outs, outs, self.train_devices
        )
        return return_outs

    def train(
        self,
        data,
        state=None,
        step=None,
        imbalanced_reward_weights=None,
        reward_values=None,
    ):
        data = data.copy()
        rng = data.pop("rng")
        if state is None:
            rng = self._next_rngs(self.train_devices)
            state, self.varibs = self._train_initial(self.varibs, rng, data["is_first"])

        prev_varibs = self.varibs
        #  Agent.train
        if step is not None:
            with jax.transfer_guard("allow"):
                # n_devices = len(self.train_devices)
                bs = data[list(data.keys())[0]].shape[0]
                step = jnp.repeat(jnp.array(step, jnp.int32), bs).reshape(bs)
                step = self._convert_inps(step, self.train_devices)

        imbalanced_reward_weights = self._convert_train_reward_weights(
            imbalanced_reward_weights
        )

        if reward_values is not None:
            with jax.transfer_guard("allow"):
                reward_values = self._convert_inps(
                    reward_values, self.train_devices, replicate=True
                )

        (outs, state, mets), self.varibs = self._train(
            self.varibs,
            rng,
            data,
            state,
            step,
            imbalanced_reward_weights,
            reward_values,
        )
        self.updates.increment()

        if not self.single_device:
            if not self.sync_promise and self.should_sync(self.updates):
                self.sync_promise = self.sync_worker.submit(
                    self._copy_varibs, prev_varibs, block=True
                )

        return_outs = self._process_train_outs(outs)

        return_mets = {}
        if self.mets_promise and self.mets_promise.done():
            return_mets = self.mets_promise.result()
            self.mets_promise = None

        if not self.mets_promise:
            # Only request metrics if we aren't currently waiting for previous  metrics. This means we'll skip the metrics of some training steps if fetching them from device would slow down the training loop.
            self.mets_promise = self.mets_worker.submit(
                self._convert_mets, mets, self.train_devices
            )

        if self.once:
            self.once = False
            assert ledwm.Optimizer.Optimizer.PARAM_COUNTS
            for name, count in ledwm.Optimizer.Optimizer.PARAM_COUNTS.items():
                if count is None:
                    cprint(f"agent.param_count_missing | name={name}", "red")
                    continue
                return_mets[f"params_{name}"] = float(count)

        if self.config_jax.profiler:
            outdir, copyto = self.logdir, None
            profile_start = int(os.environ.get("LEDWM_PROFILE_START", "100"))
            profile_steps = int(os.environ.get("LEDWM_PROFILE_STEPS", "20"))
            if profile_steps < 1:
                raise ValueError("LEDWM_PROFILE_STEPS must be at least 1")
            profile_stop = profile_start + profile_steps
            if str(outdir).startswith(("gs://", "/gcs/")):
                copyto = outdir
                outdir = Path("/tmp/profiler")
            outdir.mkdirs()
            if self.updates == profile_start:
                print(
                    f"jax.profiler_start | directory={outdir} | "
                    f"updates={profile_start}:{profile_stop}"
                )
                jax.profiler.start_trace(str(outdir))
            if self.updates == profile_stop:
                from ledwm.embodied.core import path as pathlib

                print("jax.profiler_stop")
                jax.profiler.stop_trace()
                if copyto:
                    pathlib.GFilePath(outdir).copy(copyto)
                    print(f"jax.profiler_copy | source={outdir} | destination={copyto}")

        return return_outs, state, return_mets

    def warmup_train(self, imbalanced_reward_weights=None):
        """Compile and execute the train path without changing agent state."""
        dims = (self.batch_size, self.batch_length)
        data = self._dummy_batch(self._train_spaces, dims)
        # Replay materialization normalizes symbolic IDs and positions before
        # batching. Match those input dtypes so persistent-cache keys are equal.
        data = {
            key: (
                value.astype(np.int32)
                if key.endswith("_ids") or key.endswith("_pos")
                else value
            )
            for key, value in data.items()
        }
        # Replay adds these fields after environment-space construction.
        data["reward_indicator"] = np.zeros((self.batch_size,), np.float32)
        data["sample_id"] = np.zeros((self.batch_size, 16), np.uint8)
        data = self._convert_inps(data, self.train_devices)

        seed = np.int64(0)
        if len(self.train_devices) == 1:
            rng = jax.device_put(seed, self.train_devices[0])
        else:
            rng = _device_put_replicated(seed, self.train_devices)

        state, _ = self._train_initial(self.varibs, rng, data["is_first"])
        step = None
        if self.config_jax.opt_step:
            with jax.transfer_guard("allow"):
                data_bs = data[list(data.keys())[0]].shape[0]
                step = jnp.zeros((data_bs,), jnp.int32)
                step = self._convert_inps(step, self.train_devices)

        imbalanced_reward_weights = self._convert_train_reward_weights(
            imbalanced_reward_weights
        )
        result, _ = self._train(
            self.varibs,
            rng,
            data,
            state,
            step,
            imbalanced_reward_weights,
            None,
        )
        jax.block_until_ready(result)

    def warmup_report(self):
        """Compile and execute the report path for capacity probing."""
        dims = (self.batch_size, self.batch_length)
        data = self._dummy_batch(self._train_spaces, dims)
        data = {
            key: (
                value.astype(np.int32)
                if key.endswith("_ids") or key.endswith("_pos")
                else value
            )
            for key, value in data.items()
        }
        data["reward_indicator"] = np.zeros((self.batch_size,), np.float32)
        data["sample_id"] = np.zeros((self.batch_size, 16), np.uint8)
        data = self._convert_inps(data, self.train_devices)
        data["rng"] = self._next_rngs(self.train_devices)

        step = 0 if self.config_jax.opt_step else None
        result = self.report(data, step=step)
        jax.block_until_ready(result)

    def warmup_policy(self, batch_size=None):
        """Compile and execute the actor policy path for capacity probing."""
        batch_size = int(batch_size or self.config.run.actor_batch)
        obs = self._dummy_batch(self._obs_spaces, (batch_size,))
        if "is_first" in obs:
            obs["is_first"][...] = True
        self.policy(
            obs,
            state=None,
            step=0,
            mode="train",
            return_state_to_host=False,
        )

    def _convert_train_reward_weights(self, weights):
        if weights is None:
            return None
        with jax.transfer_guard("allow"):
            n_devices = len(self.train_devices)
            weights = {
                k: jnp.repeat(jnp.array(v, jnp.float32), n_devices).reshape(
                    n_devices
                )
                for k, v in weights.items()
            }
            return self._convert_inps(weights, self.train_devices)

    def train_wm(self, data, state=None):
        data = self._convert_inps(data)
        rng = self._next_rngs(mirror=not self.varibs)
        assert state is not None
        (outs, state, mets), self.varibs = self._train(self.varibs, rng, data, state)
        outs = self._convert_outs(outs)
        mets = self._convert_mets(mets)
        return outs, state, mets

    def report_policy(
        self,
        data,
        step=None,
        # sample=False,  # argmax in policy is default
    ):
        # TODO: We could also do the same pipelining optimization used in train()
        # but it doesn't really matter because report() is not called as often.
        data = data.copy()
        rng = data.pop("rng")

        if step is not None:
            with jax.transfer_guard("allow"):
                n_devices = len(self.train_devices)
                step = jnp.zeros((n_devices,), jnp.int32)
                step = self._convert_inps(step, self.train_devices)

        mets, _ = self._report_policy(self.varibs, rng, data, step)
        mets = self._convert_mets(mets, self.train_devices)
        return mets

    def report(self, data, step=None):
        # TODO: We could also do the same pipelining optimization used in train() but it doesn't really matter because report() is not called as often.
        data = data.copy()
        rng = data.pop("rng")

        if step is not None:
            with jax.transfer_guard("allow"):
                n_devices = len(self.train_devices)
                step = jnp.zeros((n_devices,), jnp.int32)
                step = self._convert_inps(step, self.train_devices)

        mets, _ = self._report(self.varibs, rng, data, step)
        mets = self._convert_mets(mets, self.train_devices)
        return mets

    def vis(self, data, num_obs, num_imagine):
        data = data.copy()
        rng = self._next_rngs(self.train_devices)
        (recon, openl, reward), _ = self._vis(
            self.varibs, rng, data, num_obs, num_imagine
        )
        return jax.device_get(recon), jax.device_get(openl), jax.device_get(reward)

    def dataset(
        self,
        generator,
        batch_size=None,
        data_workers=None,
        prefetch_source=4,
        prefetch_batch=4,
    ):
        """
        called in parallel_learner
        dataset = agent.dataset(replay.dataset)
        """
        preprocessors = []
        if not isinstance(generator, list):
            bs = batch_size if batch_size is not None else self.batch_size
            generator = [generator] * bs

        from ledwm.embodied.core.batcher import Batcher

        batcher = Batcher(
            sources=generator,
            workers=self.data_workers if data_workers is None else data_workers,
            postprocess=self._postprocess_dataset_batch,
            prefetch_source=prefetch_source,
            prefetch_batch=prefetch_batch,
            preprocessors=self.agent.wm.encoder.preprocessors,  # empty {}
        )

        return batcher()

    def _postprocess_dataset_batch(self, batch):
        # Replay telemetry is consumed by the Python learner loop. Keeping it on
        # the host avoids a needless H2D/D2H round trip and, more importantly,
        # avoids synchronizing the prefetched training batch just to log two means.
        replay_stats = {
            key: batch.pop(key)
            for key in (REPLAY_SAMPLE_COUNT_KEY, REPLAY_SAMPLE_PRIORITY_KEY)
            if key in batch
        }
        batch = self._convert_inps(batch, self.train_devices)
        return {
            **batch,
            **replay_stats,
            "rng": self._next_rngs(self.train_devices),
        }

    def postprocess(self, batch):
        return self._postprocess_dataset_batch(batch)

    def dataload_preprocess_single(self, obs):
        """Preprocessing for dataload-time, e.g. padding elements of a batch.

        Args:
            obs: A dict with a single unbatched observation from online training
        """
        if not self.agent.preprocessors or len(self.agent.preprocessors) == 0:  # type: ignore
            return obs
        assert obs["reward"].shape == (1,)
        assert obs["language_info"].shape == (1,) and isinstance(
            obs["language_info"][0], str
        )
        obs_pp = {}
        for k, v in obs.items():
            if k in self.agent.preprocessors:  # type: ignore
                # Add batch dimension: [v] = (batch=1, time=1)
                preproc = self.agent.preprocessors[k]([v])  # type: ignore
                for preproc_key, preproc_val in preproc.items():
                    # (batch=1, time=1, tok_seq_len) -> (time=1, tok_seq_len)
                    obs_pp[f"{k}_{preproc_key}"] = preproc_val[0]
            else:
                obs_pp[k] = obs[k]
        return obs_pp

    def save(self):
        if len(self.train_devices) > 1:
            varibs = tree_map(lambda x: x[0], self.varibs)
        else:
            varibs = self.varibs
        varibs = jax.device_get(varibs)
        data = tree_map(np.asarray, varibs)
        return data

    def load(self, ck_state, load_only_key=None):
        # if self.load_wm_only:
        if self.is_loading_partial or load_only_key is not None:
            # Remove replicated dim for multiple devices in self.varibs before loading.
            if len(self.train_devices) > 1:
                orig = tree_map(lambda x: x[0], self.varibs)
            else:
                orig = self.varibs
            # Load subset of vars from checkpoint, keep orig for the rest
            cprint(
                f"agent.partial_load | configured_key={self.config.load_only_key} | "
                f"override_key={load_only_key}",
                "red",
            )
            self.varibs = jaxutils.load_partial_checkpoint(
                orig,
                ck_state,
                (
                    load_only_key
                    if load_only_key is not None
                    else self.config.load_only_key
                ),
                self.exclude_key,
            )
        else:
            if self.load_match_shape:
                assert self.varibs is not None
                expected = set(self.varibs)  # type: ignore
                found = set(ck_state)
                assert jax.tree_util.tree_structure(
                    expected
                ) == jax.tree_util.tree_structure(found), (expected, found)
                self.varibs = ck_state
            else:
                # Remove replicated dim
                if len(self.train_devices) > 1:
                    orig = tree_map(lambda x: x[0], self.varibs)
                else:
                    orig = self.varibs
                self.varibs = jaxutils.load_partial_checkpoint_shape(orig, ck_state)

        # self.varibs: {key: numpy array}
        if len(self.train_devices) == 1:
            self.varibs = jax.device_put(self.varibs, self.train_devices[0])
        else:
            self.varibs = _device_put_replicated(self.varibs, self.train_devices)

        if not self.single_device:
            print("agent.variables_sync | destination=policy_devices")
            self.policy_varibs = self._copy_varibs(self.varibs)

    def _setup(self):
        try:
            configure_tensorflow_cpp_warnings()
            import tensorflow as tf

            tf.config.set_visible_devices([], "GPU")
            tf.config.set_visible_devices([], "TPU")
        except Exception as e:
            print(f"tensorflow.device_disable_error | error={e}")

        if self.config_jax.allocator:
            os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = str(
                self.config_jax.allocator
            )
            cprint(f"jax.config | allocator={self.config_jax.allocator}", "yellow")

        if self.config_jax.quiet_xla:
            os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
            cprint("jax.config | tf_cpp_min_log_level=2", "yellow")

        if self.config_jax.prealloc:
            os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(
                self.config_jax.mem_fraction
            )
            cprint(
                f"jax.config | memory_fraction={self.config_jax.mem_fraction}",
                "yellow",
            )
        else:
            os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
            cprint(
                "jax.memory_warning | preallocate=false | risk=oom",
                "red",
            )

        xla_flags = []
        if self.config_jax.logical_cpus:
            count = self.config_jax.logical_cpus
            xla_flags.append(f"--xla_force_host_platform_device_count={count}")
        if self.config_jax.xla_autotune_level != -1:
            xla_flags.append(
                f"--xla_gpu_autotune_level={self.config_jax.xla_autotune_level}"
            )

        if xla_flags:
            existing = os.environ.get("XLA_FLAGS", "")
            os.environ["XLA_FLAGS"] = " ".join(
                part for part in [existing, *xla_flags] if part
            )
        else:
            # os.environ[
            #     "XLA_FLAGS"
            # ] = f"--xla_gpu_cuda_data_dir={self.config.cuda_data_dir}"
            # cprint(f"setting XLA_FLAGS to {os.environ['XLA_FLAGS']}", "green")
            pass

        jax.config.update("jax_platform_name", self.config_jax.platform)
        jax.config.update("jax_disable_jit", not self.config_jax.jit)
        jax.config.update("jax_debug_nans", self.config_jax.debug_nans)

        if self.config_jax.transfer_guard:
            jax.config.update("jax_transfer_guard", "disallow")

        if self.config_jax.platform == "cpu":
            jax.config.update("jax_disable_most_optimizations", self.config_jax.debug)
        jaxutils.COMPUTE_DTYPE = getattr(jnp, self.config_jax.precision)

    def _transform(self):
        self._policy_initial = nj.pure(lambda x: self.agent.policy_initial(len(x)))  # type: ignore
        self._train_initial = nj.pure(lambda x: self.agent.train_initial(len(x)))  # type: ignore
        self._policy = nj.pure(self.agent.policy)
        self._train = nj.pure(self.agent.train)
        self._finetune_policy = nj.pure(self.agent.finetune_policy)
        self._report_policy = nj.pure(self.agent.report_policy)
        self._report = nj.pure(self.agent.report)
        self._vis = nj.pure(self.agent.vis)

        if len(self.train_devices) == 1:
            kw = dict(device=self.train_devices[0])
            self._train_initial = nj.jit(self._train_initial, **kw)
            self._train = nj.jit(self._train, **kw)
            self._report = nj.jit(self._report, **kw)
            self._finetune_policy = nj.jit(self._finetune_policy, **kw)
            self._report_policy = nj.jit(self._report_policy, **kw)

        else:
            kw = dict(devices=self.train_devices)
            self._train_initial = nj.pmap(self._train_initial, "i", **kw)
            self._train = nj.pmap(self._train, "i", **kw)
            self._report = nj.pmap(self._report, "i", **kw)
            self._finetune_policy = nj.pmap(self._finetune_policy, "i", **kw)
            self._report_policy = nj.pmap(self._report_policy, "i", **kw)

        if len(self.policy_devices) == 1:
            kw = dict(device=self.policy_devices[0])
            self._policy_initial = nj.jit(self._policy_initial, **kw)
            self._policy = nj.jit(self._policy, static=["mode"], **kw)

        else:
            kw = dict(devices=self.policy_devices)
            self._policy_initial = nj.pmap(self._policy_initial, "i", **kw)
            self._policy = nj.pmap(self._policy, "i", static=["mode"], **kw)

    def _convert_inps(self, value, devices, rng=False, block=False, replicate=False):
        if replicate:
            # Replicate the value across all devices
            value = _device_put_replicated(value, devices)
        elif len(devices) == 1:
            value = jax.device_put(value, devices[0])
        else:
            check = tree_map(lambda x: len(x) % len(devices) == 0, value)
            if not all(jax.tree_util.tree_leaves(check)):
                shapes = tree_map(lambda x: x.shape, value)
                raise ValueError(
                    f"Batch must by divisible by {len(devices)} devices: {shapes}"
                )

            value = tree_map(
                lambda x: x.reshape((len(devices), -1) + x.shape[1:]), value
            )
            shards = []
            for i in range(len(devices)):
                shards.append(tree_map(lambda x: x[i], value))
            value = _device_put_sharded(shards, devices)

        if rng:
            value["rng"] = self._next_rngs(devices)
        if block:
            jax.block_until_ready(value)
        return value

    def _convert_outs(self, value, devices):
        value = jax.device_get(value)
        if len(devices) > 1:
            value = tree_map(lambda x: x.reshape((-1,) + x.shape[2:]), value)
        return value

    # use this as standalone function
    def _convert_mets(self, value, devices):
        if len(devices) > 1:
            value = tree_map(lambda x: x[0], value)
        return jax.device_get(value)

    def _next_rngs(self, devices, mirror=False, high=2**63 - 1):
        if len(devices) == 1:
            return jax.device_put(self.rng.integers(high), devices[0])
        elif mirror:
            return _device_put_replicated(self.rng.integers(high), devices)
        else:
            return _device_put_sharded(
                list(self.rng.integers(high, size=len(devices))), devices
            )

    def _init_varibs(self, obs_space, act_space):
        varibs = {}
        rng = self._next_rngs(self.train_devices, mirror=True)
        dims = (self.batch_size, self.batch_length)
        data = self._dummy_batch({**obs_space, **act_space}, dims)
        data = self._convert_inps(data, self.train_devices)

        # call agent.train_initial:
        # state = prev_latent, prev_action;
        # varibs = ?
        # init bs = len(data['is_first'])
        # type: Callable[Agent.train_initial]
        state, varibs = self._train_initial(varibs, rng, data["is_first"])

        # return outs, state, metrics
        step = None
        if self.config_jax.opt_step:
            with jax.transfer_guard("allow"):
                # n_devices = len(self.train_devices)
                data_bs = data[list(data.keys())[0]].shape[0]
                step = jnp.zeros((data_bs,), jnp.int32)
                step = self._convert_inps(step, self.train_devices)

        varibs = self._train(varibs, rng, data, state, step=step, init_only=True)
        return varibs

    def _copy_varibs(self, varibs, block=False):
        if self.single_device:
            return varibs

        # varibs = jax.block_until_ready(varibs)
        # with jax.transfer_guard("allow"):
        #     if len(self.policy_devices) == 1:
        #         varibs = jax.device_put(varibs, self.policy_devices[0])
        #     else:
        #         varibs = jax.device_put_replicated(varibs, self.policy_devices)

        if len(self.train_devices) > 1:
            varibs = tree_map(_unreplicate_for_policy, varibs)

        if len(self.policy_devices) == 1:
            varibs = jax.device_put(varibs, self.policy_devices[0])
        else:
            varibs = _device_put_replicated(varibs, self.policy_devices)

        if block:
            jax.block_until_ready(varibs)
        return varibs

    def _dummy_batch(self, spaces, batch_dims):
        spaces = [(k, v) for k, v in spaces.items() if not k.startswith("log_")]
        data = {k: np.zeros(v.shape, v.dtype) for k, v in spaces}
        for dim in reversed(batch_dims):
            data = {k: np.repeat(v[None], dim, axis=0) for k, v in data.items()}
        return data
