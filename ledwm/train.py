import importlib
import os
import pathlib
import pickle
import shlex
import shutil
import sys
import warnings
from collections import OrderedDict
from functools import partial as bind
import datetime

# Running this file by path makes Python put ``ledwm/`` rather than the
# repository root on sys.path. Bootstrap the local package roots before the
# first project import so every launcher works without relying on PYTHONPATH.
_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
for _PACKAGE_ROOT in (_REPO_ROOT, _REPO_ROOT / "messenger-emma"):
    if str(_PACKAGE_ROOT) not in sys.path:
        sys.path.insert(0, str(_PACKAGE_ROOT))

try:
    from ledwm.startup import (
        configure_numeric_threading,
        configure_jax_compilation_cache,
        configure_tensorflow_cpp_warnings,
        should_announce_jax_cache,
    )
except ModuleNotFoundError:  # Support `python ledwm/train.py`.
    from startup import (
        configure_numeric_threading,
        configure_jax_compilation_cache,
        configure_tensorflow_cpp_warnings,
        should_announce_jax_cache,
    )

try:
    from ledwm.logging_setup import (
        complete_logging,
        configure_logging,
        logger as event_logger,
    )
except ModuleNotFoundError:  # Support `python ledwm/train.py`.
    from logging_setup import (
        complete_logging,
        configure_logging,
        logger as event_logger,
    )


configure_numeric_threading()
configure_tensorflow_cpp_warnings()
configure_logging()

DEFAULT_EARLY_JAX_ALLOCATOR = "platform"
DEFAULT_EARLY_JAX_PREALLOCATE = "false"
DEFAULT_EARLY_JAX_XLA_AUTOTUNE_LEVEL = "4"


def _append_xla_flag(flag):
    existing = os.environ.get("XLA_FLAGS", "")
    if existing and flag in shlex.split(existing):
        return
    parts = [existing, flag] if existing else [flag]
    os.environ["XLA_FLAGS"] = " ".join(part for part in parts if part)


def _has_xla_flag(name):
    existing = os.environ.get("XLA_FLAGS", "")
    if not existing:
        return False
    return any(
        part == name or part.startswith(f"{name}=") for part in shlex.split(existing)
    )


def _sanitize_xla_flags():
    # JAX 0.6.2 can fail some report-time small dot_general shapes with:
    # "Too small divisible part of the contracting dimension" when deterministic
    # GPU XLA ops are enabled. Do this before importing JAX.
    existing = os.environ.get("XLA_FLAGS")
    if not existing:
        return
    parts = shlex.split(existing)
    kept = [
        part
        for part in parts
        if not (
            part == "--xla_gpu_deterministic_ops"
            or part == "--xla_gpu_deterministic_ops=true"
        )
    ]
    if kept != parts:
        os.environ["XLA_FLAGS"] = " ".join(kept)
        event_logger.warning(
            "jax.xla_flags | removed=--xla_gpu_deterministic_ops | "
            "reason=avoid_dot_general_failures"
        )


def _early_configure_jax_allocator(argv):
    _sanitize_xla_flags()
    mem_fraction = None
    prealloc = None
    allocator = None
    xla_autotune_level = None
    quiet_xla = None
    for index, arg in enumerate(argv):
        if arg.startswith("--jax.mem_fraction="):
            mem_fraction = arg.split("=", 1)[1]
        elif arg == "--jax.mem_fraction" and index + 1 < len(argv):
            mem_fraction = argv[index + 1]
        elif arg.startswith("--jax.prealloc="):
            prealloc = arg.split("=", 1)[1]
        elif arg == "--jax.prealloc" and index + 1 < len(argv):
            prealloc = argv[index + 1]
        elif arg.startswith("--jax.allocator="):
            allocator = arg.split("=", 1)[1]
        elif arg == "--jax.allocator" and index + 1 < len(argv):
            allocator = argv[index + 1]
        elif arg.startswith("--jax.xla_autotune_level="):
            xla_autotune_level = arg.split("=", 1)[1]
        elif arg == "--jax.xla_autotune_level" and index + 1 < len(argv):
            xla_autotune_level = argv[index + 1]
        elif arg.startswith("--jax.quiet_xla="):
            quiet_xla = arg.split("=", 1)[1]
        elif arg == "--jax.quiet_xla" and index + 1 < len(argv):
            quiet_xla = argv[index + 1]

    if allocator is None:
        allocator = os.environ.get("XLA_PYTHON_CLIENT_ALLOCATOR")
    if allocator is None:
        allocator = DEFAULT_EARLY_JAX_ALLOCATOR
    if prealloc is None:
        prealloc = os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE")
    if prealloc is None:
        prealloc = DEFAULT_EARLY_JAX_PREALLOCATE

    if mem_fraction is not None:
        os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = mem_fraction
    if prealloc is not None and prealloc.lower() in ("false", "0", "no"):
        os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    elif prealloc is not None and prealloc.lower() in ("true", "1", "yes"):
        os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "true"
    if allocator:
        os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = allocator
    if allocator and allocator.lower() == "platform":
        if xla_autotune_level is None and not _has_xla_flag(
            "--xla_gpu_autotune_level"
        ):
            xla_autotune_level = DEFAULT_EARLY_JAX_XLA_AUTOTUNE_LEVEL
        if quiet_xla is None:
            quiet_xla = "true"
    if xla_autotune_level is not None and xla_autotune_level != "-1":
        _append_xla_flag(f"--xla_gpu_autotune_level={xla_autotune_level}")
        # With autotuning disabled (level 0, the default for the platform
        # allocator) XLA cannot fall back off its Triton GEMM emitter, which
        # rejects some model matmul shapes at compile time with
        # "CANCELLED: Too small divisible part of the contracting dimension".
        # Route GEMMs to cuBLAS instead: heuristic-selected, fast, and needs no
        # autotuning. Skip if the user already pinned the flag themselves.
        if xla_autotune_level == "0" and "triton_gemm" not in os.environ.get(
            "XLA_FLAGS", ""
        ):
            _append_xla_flag("--xla_gpu_enable_triton_gemm=false")
    if quiet_xla is not None and quiet_xla.lower() in ("true", "1", "yes"):
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


_early_configure_jax_allocator(sys.argv[1:])

import numpy as np
from termcolor import cprint

from ledwm.embodied.core.config import Config
from ledwm.embodied.core.counter import Counter
from ledwm.embodied.core.flags import Flags
from ledwm.embodied.core.logger import Logger
from ledwm.embodied.core.path import Path
from ledwm.embodied.core.JSONLOutput import JSONLOutput
from ledwm.embodied.core.run_output_logs import (
    install_run_output_logs,
    save_run_log_artifact,
)
from ledwm.embodied.core.parallel import Parallel
from ledwm.embodied.core.batch_env import BatchEnv
from ledwm.embodied.replay.replays import make_replay
from ledwm.embodied.core import wrappers
from ledwm.configs_util import apply_named_config
from ledwm.resume import LATEST, latest_resumable_run, normalize_resume_flag

os.environ["WANDB_CACHE_DIR"] = "./wandb/cache"
os.environ["WANDB_DATA_DIR"] = "./wandb/data"
pathlib.Path(os.environ["WANDB_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)
pathlib.Path(os.environ["WANDB_DATA_DIR"]).mkdir(parents=True, exist_ok=True)

# sys.path append parent of current file
sys.path.append(str(pathlib.Path(__file__).parent.parent))

warnings.filterwarnings("ignore", ".*box bound precision lowered.*")
warnings.filterwarnings("ignore", ".*using stateful random seeds*")
warnings.filterwarnings("ignore", ".*is a deprecated alias for.*")
warnings.filterwarnings("ignore", ".*truncated to dtype int32.*")

directory = pathlib.Path(__file__).resolve()
directory = directory.parent
sys.path.append(str(directory.parent))
sys.path.append(str(directory.parent.parent))
sys.path.append(str(directory.parent.parent.parent))


__package__ = directory.name


def custom_repr(self):
    return f"Array:{tuple(self.shape)}"


_jax_module = None
_wandb_module = None
_agt_module = None


def _enable_persistent_compilation_cache(jax_module):
    """Reuse XLA-compiled binaries across process restarts.

    On the first run this compiles as usual and writes the compiled
    executables to disk; subsequent runs with the same JAX/jaxlib version,
    XLA flags, and input shapes load them in seconds instead of recompiling
    the whole world model + actor-critic from scratch.

    Disable by setting JAX_COMPILATION_CACHE_ENABLE=0. Override the location
    with JAX_COMPILATION_CACHE_DIR (default: ~/.cache/ledwm_jax_cache).
    """
    if os.environ.get("JAX_COMPILATION_CACHE_ENABLE", "1").lower() in (
        "0",
        "false",
        "no",
    ):
        return
    cache_dir = os.environ.get(
        "JAX_COMPILATION_CACHE_DIR",
        os.path.expanduser("~/.cache/ledwm_jax_cache"),
    )
    try:
        if configure_jax_compilation_cache(jax_module) and should_announce_jax_cache():
            event_logger.info(
                "jax.compilation_cache | enabled=true | directory={}", cache_dir
            )
    except Exception as exc:  # pragma: no cover - never block training on cache setup
        if should_announce_jax_cache():
            event_logger.warning(
                "jax.compilation_cache | enabled=false | error={} | "
                "action=recompile",
                exc,
            )


def _get_jax():
    """Import JAX only in the real trainer, not in spawned env workers."""
    global _jax_module
    if _jax_module is None:
        import jax as jax_module

        _jax_module = jax_module
        _enable_persistent_compilation_cache(_jax_module)
        _jax_module.Array.__repr__ = custom_repr
    return _jax_module


def _get_wandb():
    global _wandb_module
    if _wandb_module is None:
        import wandb as wandb_module

        _wandb_module = wandb_module
    return _wandb_module


def _get_agent_module():
    """Lazy-load ledwm.agent to avoid heavy import at startup."""
    global _agt_module
    if _agt_module is None:
        from . import agent as agt

        _agt_module = agt
    return _agt_module

original_repr = np.ndarray.__repr__


# Define the new custom repr function
def np_custom_repr(self):
    return f"{{Array:{tuple(self.shape)}}} {original_repr(self)}"


np.array2string = np_custom_repr


def random_act(env, reset=False):
    act = {k: v.sample() for k, v in env.act_space.items()}
    act["reset"] = reset
    return act


def test_obs(obs):
    for k, v in obs.items():
        if isinstance(v, np.ndarray):
            print(k, " ", v.shape, v.dtype)
        else:
            print(k, v)


def test_env(env):
    test_obs(env.step(random_act(env, reset=True)))
    print("")
    test_obs(env.step(random_act(env)))


def get_logdir(config):
    if config.logdir == "":
        from ledwm.embodied.core.WandBOutput import get_logdir_from_config

        config = config.update({"logdir": get_logdir_from_config(config)})

    if config.run.resume == LATEST:
        checkpoint_name = (
            "checkpoint.ckpt"
            if "offline_wm" in config.run.script
            else "checkpoint_1.ckpt"
        )
        res = latest_resumable_run(config.logdir, checkpoint_name)
        cprint(
            f"run.resume_latest | root={config.logdir} | selected={res} | "
            f"checkpoint={checkpoint_name} | replay=episodes/*.npz",
            "green",
        )
    elif config.run.resume != "":
        res = pathlib.Path(config.logdir) / config.run.resume
    else:
        time_str = datetime.datetime.now().strftime("%m-%d#%H-%M-%S")
        time_str += f"#{np.random.randint(1000)}"
        res = pathlib.Path(config.logdir) / f"{config.seed}@{time_str}"

    cprint(f"run.logdir | path={res}", "green")
    return res


def add_path_if_just_dir_name(ckpt_path, logdir, config):
    if ckpt_path != "" and "checkpoint" not in ckpt_path:
        checkpoint_name = (
            "checkpoint.ckpt"
            if "offline_wm" in config.run.script
            else "checkpoint_1.ckpt"
        )
        res = pathlib.Path(logdir).parent / ckpt_path / checkpoint_name
        cprint(f"checkpoint.resolve | path={res}", "green")
        return res
    else:
        return ckpt_path


def configure_jax_allocator(config):
    if config.jax.allocator:
        os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = str(config.jax.allocator)
        cprint(f"jax.config | allocator={config.jax.allocator}", "yellow")
    # Mirror _early_configure_jax_allocator so a bare run (flags coming from the
    # config defaults rather than the CLI) resolves to the same XLA flags. When
    # autotuning is explicitly disabled, XLA must be kept off its Triton GEMM
    # emitter (which rejects some model matmul shapes with "Too small divisible
    # part of the contracting dimension") by routing GEMMs to cuBLAS.
    autotune_level = config.jax.xla_autotune_level
    if config.jax.allocator == "platform":
        if autotune_level == -1:
            autotune_level = 0
        if not config.jax.quiet_xla:
            os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    if autotune_level != -1:
        _append_xla_flag(f"--xla_gpu_autotune_level={autotune_level}")
        cprint(f"jax.config | xla_gpu_autotune_level={autotune_level}", "yellow")
        if autotune_level == 0 and "triton_gemm" not in os.environ.get(
            "XLA_FLAGS", ""
        ):
            _append_xla_flag("--xla_gpu_enable_triton_gemm=false")
            cprint(
                "jax.config | triton_gemm=false | gemm_backend=cublas", "yellow"
            )
    if config.jax.quiet_xla:
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
        cprint("jax.config | tf_cpp_min_log_level=2", "yellow")
    env_mem_fraction = os.environ.get("XLA_PYTHON_CLIENT_MEM_FRACTION")
    env_prealloc = os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE", "")
    prealloc_false = env_prealloc.lower() in ("false", "0", "no")
    if prealloc_false:
        os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
        cprint(
            "jax.memory_warning | preallocate=false | risk=oom",
            "red",
        )
    elif config.jax.prealloc or env_mem_fraction is not None:
        mem_fraction = env_mem_fraction or str(config.jax.mem_fraction)
        os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(mem_fraction)
        cprint(f"jax.config | memory_fraction={mem_fraction}", "yellow")
    else:
        os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
        cprint(
            "jax.memory_warning | preallocate=false | risk=oom",
            "red",
        )


def configure_training_speed_flags(config):
    """Resolve task-configured lean training flags for every launch path."""
    settings = {
        "LEDWM_FAST_TRAIN_METRICS": config.run.fast_train_metrics,
        "LEDWM_FAST_OPTIMIZER_METRICS": config.run.fast_optimizer_metrics,
        "LEDWM_SKIP_ADAM_METRICS": config.run.skip_adam_metrics,
        "LEDWM_SKIP_TRAIN_OUTS": config.run.skip_train_outs,
    }
    for name, configured in settings.items():
        source = "environment" if name in os.environ else "config"
        os.environ.setdefault(name, "1" if configured else "0")
        cprint(
            f"train.speed_flag | name={name} | value={os.environ[name]} | "
            f"source={source}",
            "yellow",
        )


def warmup_agent_for_capacity_probe(
    agent, args, reward_weights=None, include_policy=True
):
    """Exercise every enabled GPU path before accepting an auto batch size."""
    if include_policy:
        agent.warmup_policy(args.actor_batch)
    agent.warmup_train(imbalanced_reward_weights=reward_weights)
    if args.report.train:
        agent.warmup_report()


def _validate_jax_device_selection(config, devices):
    num_devices = len(devices)
    valid = set(range(num_devices))
    selections = {
        "policy_devices": tuple(config.jax.policy_devices),
        "train_devices": tuple(config.jax.train_devices),
    }
    for name, selected in selections.items():
        assert selected, f"jax.{name} must include at least one device"
        assert len(set(selected)) == len(selected), (
            f"jax.{name} contains duplicate device indices: {selected}"
        )
        invalid = [index for index in selected if index not in valid]
        assert not invalid, (
            f"jax.{name}={selected} must use visible JAX device indices "
            f"0..{num_devices - 1}; invalid={invalid}"
        )


def _resolve_train_batch_size(batch_size, train_devices):
    train_count = len(train_devices)
    resolved = int(batch_size) - (int(batch_size) % train_count)
    assert resolved >= train_count, (
        f"batch_size={batch_size} is too small for {train_count} train devices; "
        f"use at least {train_count}"
    )
    return resolved


def create_args(config):
    return Config(
        **config.run,
        logdir=f"{config.logdir}",
        batch_steps=config.batch_size * config.batch_length,
    )


OFFLINE_WM_ROOT = pathlib.Path("logdir/lwm/offline_wm")


def get_offline_wm_dir(key, config, mode="hard"):
    assert key in ["train", "test"], key
    res = (
        OFFLINE_WM_ROOT
        / key
        / f"disappear={config.env.lwm.disappear}"
        / f"movement={config.use_movement_class}"
    )
    if config.offline_one_config:
        res = res / f"one_config={config.offline_one_config}"

    if key == "test":
        return res / mode
    else:
        return res


def create_offline_wm_data(logger, config, logdir, cleanup):
    from ledwm.embodied.run.create_oracle_wm_data import DataGenerator

    assert config.oracle_data_split in ["all", "train", "test"], (
        config.oracle_data_split
    )

    if config.oracle_data_split == "all" or config.oracle_data_split == "train":
        # delete train relay
        dir = get_offline_wm_dir("train", config)
        if dir.exists():
            cprint(f"dataset.delete | split=train | directory={dir}", "red")
            shutil.rmtree(dir)

        train_replay = make_replay(config, dir)
        train_env = wrapped_env(config, batch=False)
        # assert isinstance(train_env, FromGym), "train_env must be a FromGym"
        cleanup.append(train_env)
        train_data_generator = DataGenerator(
            train_env,  # type: ignore[arg-type]
            train_replay,  # type: ignore[arg-type]
            logger,
            config,
        )
        train_data_generator.generate_data()
        cprint(f"dataset.generate_done | split=train | directory={dir}", "green")
        if config.offline_one_config:
            # write movement_role_configs to file
            with open(dir / "movement_role_configs.pkl", "wb") as f:
                pickle.dump(train_data_generator.movement_role_configs, f)
                movement_role_configs_path = dir / "movement_role_configs.pkl"
                print(
                    "dataset.config_written | "
                    f"path={movement_role_configs_path}"
                )

    if config.oracle_data_split == "all" or config.oracle_data_split == "test":
        from ledwm.embodied.core.OracleAgent import INTENTIONS

        for split in ["easy", "medium", "hard"]:
            dir = get_offline_wm_dir("test", config, split)
            if dir.exists():
                cprint(f"dataset.delete | split={split} | directory={dir}", "red")
                shutil.rmtree(dir)

            test_replay = make_replay(config, dir)
            # change config to test
            test_config = config.update({"env": {"lwm": {"mode": "test"}}})
            test_env = wrapped_env(test_config, batch=False)
            # assert isinstance(test_env, FromGym), "test_env must be a FromGym"
            cleanup.append(test_env)
            test_data_generator = DataGenerator(
                test_env,  # type: ignore[arg-type]
                test_replay,  # type: ignore[arg-type]
                logger,
                test_config,
                split=split,
                num_repeat=len(INTENTIONS),
                behavior_policy="test",
            )
            if config.offline_one_config:
                # read movement_role_configs from file
                with open(movement_role_configs_path, "rb") as f:
                    movement_role_configs = pickle.load(f)
                test_data_generator.generate_data(movement_role_configs)
            else:
                test_data_generator.generate_data()

            cprint(
                f"dataset.generate_done | split={split} | directory={dir}",
                "green",
            )


def set_report_bs_bl(config):
    config = config.update(
        {
            "run": {
                "report": {
                    "first_bl": config.batch_length,
                    "first_bs": config.batch_size,
                }
            },
        }
    )
    return config


def run_train_offline_wm(
    logger,
    config,
    train_replay_dir,
    eval_replay_dir=None,
    cleanup=None,
):
    from ledwm.embodied.run.offline import train_offline_wm

    config = set_report_bs_bl(config)
    train_replay_dir = get_offline_wm_dir(config.train_offline_on, config)
    train_replay = make_replay(config, train_replay_dir, rate_limit=False)
    train_env = wrapped_env(config, batch=False)
    if cleanup is not None:
        cleanup.append(train_env)

    assert config.oracle_data_mode in ["easy", "medium", "hard"], (
        config.oracle_data_mode
    )
    if eval_replay_dir is None:
        eval_replay_dir = get_offline_wm_dir("test", config, config.oracle_data_mode)
    eval_replay = make_replay(config, eval_replay_dir, rate_limit=False)

    agent = _get_agent_module().Agent(
        train_env.obs_space,
        train_env.act_space,
        step=None,
        config=config,
        env_cache=train_env.env_cache,
    )

    train_offline_wm(agent, train_replay, eval_replay, logger, config)  # type: ignore[arg-type]


def run_eval_offline_wm(logger, config, cleanup):
    from ledwm.embodied.run.offline import eval_offline_wm

    config = set_report_bs_bl(config)
    assert config.oracle_data_mode in ["easy", "medium", "hard"], (
        config.oracle_data_mode
    )
    eval_dir = get_offline_wm_dir("test", config, config.oracle_data_mode)
    eval_replay = make_replay(
        config,
        eval_dir,
        rate_limit=False,
    )
    eval_env = wrapped_env(config, batch=False)
    # assert isinstance(eval_env, FromGym), "eval_env must be a FromGym"
    cleanup.append(eval_env)
    agent = _get_agent_module().Agent(
        eval_env.obs_space,
        eval_env.act_space,
        step=None,
        config=config,
        env_cache=eval_env.env_cache,
    )
    eval_offline_wm(agent, eval_replay, logger, config)  # type: ignore[arg-type]


def run_train(logger, config, args, logdir, step, cleanup):
    from ledwm.embodied.run.train import train

    train_replay = make_replay(config, logdir / "episodes", rate_limit=True)
    train_env = wrapped_env(
        config,
        True,
    )
    cleanup.append(train_env)
    agent = _get_agent_module().Agent(
        train_env.obs_space,
        train_env.act_space,
        step,
        config,
        train_env.env_cache,
    )
    train(agent, train_env, train_replay, logger, args)


def run_finetune_policy_seen_traj(logger, config, args, logdir, cleanup):
    train_replay = make_replay(config, logdir / "episodes", rate_limit=False)
    train_env = wrapped_env(
        config,
        True,
    )
    cleanup.append(train_env)
    agent = _get_agent_module().Agent(
        train_env.obs_space,
        train_env.act_space,
        Counter(),
        config,
        train_env.env_cache,
    )
    from ledwm.embodied.run.train import train

    train(agent, train_env, train_replay, logger, args)


def tune_finetune_policy(logger, config, logdir, cleanup):
    wandb = _get_wandb()
    sweep_configuration = {
        "method": "random",
        "metric": {"goal": "maximize", "name": "after_diff_means"},
        "parameters": {
            "critic_lr": {"max": 1e-4, "min": 1e-6},
            "step_finetune": {"min": 100, "max": 10000},
        },
    }
    sweep_id = wandb.sweep(sweep=sweep_configuration, project=config.task)
    wandb.agent(
        sweep_id,
        lambda: run_tune_finetune_policy(logger, config, logdir, cleanup),
        count=100,
    )


def run_tune_finetune_policy(logger, config, logdir, cleanup):
    wandb = _get_wandb()
    logger = make_logger(None, logdir, Counter(), config)
    config = config.update(
        {
            "critic_opt": {"lr": wandb.config.critic_lr},
            "actor_opt": {
                "lr": wandb.config.critic_lr * 2
            },  # lr is smaller -> actor is slower to change
            "run": {"step_finetune": wandb.config.step_finetune},
        }
    )
    args = config.run
    assert config.run.actor_batch <= config.envs.amount, (
        config.run.actor_batch,
        config.envs.amount,
    )
    assert args.actor_batch <= config.envs.amount, (
        args.actor_batch,
        config.envs.amount,
    )
    step = Counter()
    eval_env = wrapped_env(
        config,
        batch=True,  # match driver and driver_finetune
        # save_sent_emb=True,
    )
    agent = _get_agent_module().Agent(
        eval_env.obs_space,
        eval_env.act_space,
        step,
        config,
        eval_env.env_cache,  # type: EncoderEmbed.sent_embed
    )
    assert logger is not None, logger
    assert eval_env.env_cache is not None, eval_env
    from ledwm.compute_on_fixed_wm import compute_on_fixed_wm

    compute_on_fixed_wm(
        agent,
        eval_env,
        logger,
        config,
    )
    cleanup.append(eval_env)
    # env.close()


def run_compute_on_fixed_wm(logger, config, args, logdir, cleanup):
    from ledwm.compute_on_fixed_wm import compute_on_fixed_wm

    step = Counter()
    eval_env = wrapped_env(
        config,
        batch=True,  # match driver and driver_finetune
        # save_sent_emb=True,
    )
    agent = _get_agent_module().Agent(
        eval_env.obs_space,
        eval_env.act_space,
        step,
        config,
        eval_env.env_cache,  # type: EncoderEmbed.sent_embed
    )
    assert eval_env.env_cache is not None, eval_env
    compute_on_fixed_wm(
        agent,
        eval_env,
        # replay,
        logger,
        # args,
        config,
    )
    cleanup.append(eval_env)
    # env.close()


def run_parallel_train_evaluation(logger, config, args, logdir):
    assert config.run.actor_batch <= config.envs.amount, (
        config.run.actor_batch,
        config.envs.amount,
    )
    assert args.actor_batch <= config.envs.amount, (
        args.actor_batch,
        config.envs.amount,
    )
    step = Counter()

    train_ctor = bind(
        wrapped_env,
        config,
        batch=False,
        # save_sent_emb=True,
    )
    train_env = train_ctor()
    eval_ctor = bind(wrapped_env, config, batch=False, mode="eval")
    if "test" in args.script:
        test_ctor = bind(wrapped_env, config, batch=False, mode=config.test_set)
    else:
        test_ctor = None

    agent = _get_agent_module().Agent(
        train_env.obs_space,
        train_env.act_space,
        step,
        config,
        train_env.env_cache,  # type: EncoderEmbed.sent_embed,
        # REWARD_VALUES,
    )
    train_env.close()

    if getattr(args, "compile_only", False):
        from ledwm.train_signature import canonical_reward_weight_keys

        reward_keys = canonical_reward_weight_keys(config)
        if config.replay.imbalance == "balanced_weight" and reward_keys is None:
            raise ValueError(
                f"compile-only reward signature is not defined for task={config.task}"
            )
        weights = (
            OrderedDict((key, 1.0) for key in reward_keys)
            if reward_keys is not None
            else None
        )
        start = datetime.datetime.now()
        include_policy = os.environ.get(
            "LEDWM_CAPACITY_PROBE_POLICY", "1"
        ).lower() not in ("0", "false", "no")
        cprint(
            f"run.compile_only | state=started | batch_size={config.batch_size} | "
            f"batch_length={config.batch_length} | reward_keys={reward_keys} | "
            f"policy={str(include_policy).lower()}",
            "yellow",
        )
        warmup_agent_for_capacity_probe(
            agent, args, weights, include_policy=include_policy
        )
        elapsed = (datetime.datetime.now() - start).total_seconds()
        cprint(
            f"run.compile_only | state=complete | elapsed={elapsed:.3f}s | "
            f"cache={os.environ.get('JAX_COMPILATION_CACHE_DIR', '')}",
            "green",
        )
        return

    train_replay = make_replay(config, logdir / "episodes", rate_limit=True)
    eval_replay = make_replay(
        config,
        None,
        is_eval=True,
        batch_length=config.run.report.first_bl,
        batch_size=config.run.report.first_bs,
        rate_limit=False,
    )
    if "test" in args.script:
        test_replay = make_replay(
            config,
            None,
            is_eval=True,
            batch_length=config.run.report.first_bl,
            batch_size=config.run.report.first_bs,
            rate_limit=False,
        )
    else:
        test_replay = None

    assert logger is not None, logger
    from ledwm.embodied.run.smoothing import ReplayEps

    assert isinstance(eval_replay, ReplayEps) or eval_replay is None, type(eval_replay)
    assert isinstance(test_replay, ReplayEps) or test_replay is None, type(test_replay)
    from ledwm.embodied.run.parallel import parallel

    parallel(
        agent,
        train_replay,
        logger,
        train_ctor,
        # config.envs.amount,
        args,
        train_env.env_cache,
        eval_ctor,
        test_ctor,
        eval_replay=eval_replay,
        test_replay=test_replay,
        config=config,
    )


def run_finetune_wm(logger, config, args, logdir, cleanup):
    assert config.run.actor_batch <= config.envs.amount, (
        config.run.actor_batch,
        config.envs.amount,
    )
    assert args.actor_batch <= config.envs.amount, (
        args.actor_batch,
        config.envs.amount,
    )
    step = Counter()

    train_ctor = bind(
        wrapped_env,
        config,
        batch=False,
    )
    train_env = train_ctor()

    agent = _get_agent_module().Agent(
        train_env.obs_space,
        train_env.act_space,
        step,
        config,
        train_env.env_cache,  # type: EncoderEmbed.sent_embed,
        # REWARD_VALUES,
    )
    train_env.close()

    train_replay = make_replay(config, logdir / "episodes", rate_limit=True)

    assert logger is not None, logger
    assert train_env.env_cache is not None, train_env
    from ledwm.embodied.run.parallel import parallel

    parallel(
        agent,
        train_replay,
        logger,
        train_ctor,
        args,
        train_env.env_cache,
        config=config,
    )  # type: ignore


def run_parallel(logger, config, args, logdir):
    assert config.run.actor_batch <= config.envs.amount, (
        config.run.actor_batch,
        config.envs.amount,
    )
    assert args.actor_batch <= config.envs.amount, (
        args.actor_batch,
        config.envs.amount,
    )
    train_ctor = bind(wrapped_env, config, batch=False)
    step = Counter()
    train_env = train_ctor()
    agent = _get_agent_module().Agent(
        train_env.obs_space,
        train_env.act_space,
        step,
        config,
        train_env.env_cache,
    )
    train_env.close()
    train_replay = make_replay(config, logdir / "episodes", rate_limit=True)
    assert logger is not None, logger
    from ledwm.embodied.run.parallel import parallel

    parallel(
        agent,
        train_replay,
        logger,
        train_ctor,
        args,
        config=config,
    )


def run_parallel_eval(logger, config, args):
    assert logger is not None
    assert config.run.actor_batch <= config.envs.amount, (
        config.run.actor_batch,
        config.envs.amount,
    )
    ctor = bind(
        wrapped_env,
        config,
        batch=False,
        # save_sent_emb=True,
        # test_sent_emb_only=True,
    )
    step = Counter()
    env = ctor()
    agent = _get_agent_module().Agent(
        env.obs_space,
        env.act_space,
        step,
        config,
        env.env_cache,
    )
    env.close()
    ctor = bind(
        wrapped_env,
        config,
        batch=False,
        # save_sent_emb=False,
        # test_sent_emb_only=True,
    )
    from ledwm.embodied.run.parallel_eval import parallel_eval

    parallel_eval(agent, logger, ctor, config.envs.amount, args, config)


def update_config(config):
    jax = _get_jax()
    if config.drop_x_randomly_in_query:
        # assert not train
        assert "train" in config.run.script, (
            "drop_x_randomly_in_query must be True when train"
        )

    # if not deter_game then assert decay_multi_step is True
    if "train" in config.run.script:
        if not config.env.messenger.deter_game and "s1" not in config.task:
            assert config.decay_multi_step, "decay_multi_step must be True"
        else:
            assert not config.decay_multi_step, "decay_multi_step must be False"

    # Honor the configured seed. Persistent XLA cache keys can include traced
    # constants derived from it, so silently randomizing it defeats reuse.
    cprint(f"run.seed | value={config.seed} | source=config")

    config = config.update({"logdir": get_logdir(config)})
    if config.run.resume != "":
        config = config.update(
            {"replay.resume": True, "run.load_checkpoint": True}
        )

    if config.run.debug:
        jax.config.update("jax_debug_nans", True)

    # if not config.run.debug and config.run.train_ratio < 64:
    #     raise ValueError("train_ratio must be greater than 64")

    if not config.run.debug and config.envs.amount == 0:
        assert isinstance(config.run.train_ratio, int)
        config = config.update(
            {"envs": {"amount": int(512 / config.run.train_ratio * 4)}}
        )
        cprint(f"run.env_count | value={config.envs.amount} | source=auto")
        cprint(
            f"run.actor_batch | value={config.envs.amount} | source=env_count",
            "red",
        )
    from ledwm.embodied.core.distr import find_random_port

    # actor_batch defaults to envs.amount (sentinel 0). Set --run.actor_batch >0
    # to fire the actor callback on a subset of envs, so replay starts filling
    # before all env workers are up and stragglers don't stall each step.
    resolved_actor_batch = config.run.actor_batch or config.envs.amount
    cprint(
        f"run.actor_batch | value={resolved_actor_batch} | "
        f"env_count={config.envs.amount}",
        "red",
    )
    config = config.update(
        {
            "run": {
                "actor_batch": resolved_actor_batch,
                "actor_port": find_random_port(),
                "from_checkpoint": add_path_if_just_dir_name(
                    config.run.from_checkpoint, config.logdir, config
                ),
            }
        }
    )
    # resume replay when from_checkpoint
    if (
        config.run.from_checkpoint != "" and config.resume_replay
    ) or config.run.resume_replay_from != "":
        config = config.update({"replay.resume": True})

    # assert isinstance(config, Config)
    assert config.batch_size >= len(config.jax.policy_devices), (
        f"{config.batch_size} must > {len(config.jax.policy_devices)}"
    )

    devices = jax.devices()
    cprint(f"jax.devices | count={len(devices)} | devices={devices}", "green")
    try:
        stats = devices[0].memory_stats()
        if stats:
            cprint(f"jax.memory | device=0 | stats={stats}", "yellow")
    except Exception as e:
        cprint(f"jax.memory_unavailable | device=0 | error={e}", "yellow")
    _validate_jax_device_selection(config, devices)
    resolved_batch_size = _resolve_train_batch_size(
        config.batch_size, config.jax.train_devices
    )
    if resolved_batch_size != config.batch_size:
        cprint(
            f"run.batch_size_adjusted | requested={config.batch_size} | "
            f"train_devices={len(config.jax.train_devices)} | "
            f"effective={resolved_batch_size}",
            "yellow",
        )
        config = config.update({"batch_size": resolved_batch_size})

    assert pathlib.Path(str(config.run.from_checkpoint)).exists(), (
        f"{config.run.from_checkpoint} does not exist"
    )

    if config.run.script == "train_eval":
        config = config.update({"env": {"message": {"mode": "train_eval"}}})

    return config


def run_eval_only():
    pass
    # train_replay = make_replay(config, logdir / "episodes")
    # eval_replay = make_replay(config, logdir / "eval_episodes", is_eval=True)
    # train_env = wrapped_env(config, batch=True)
    # eval_env = wrapped_env(config, batch=True)
    # cleanup += [train_env, eval_env]
    # agent = agt.Agent(
    #     train_env.obs_space,
    #     train_env.act_space,
    #     step,
    #     config,
    #     train_env.env_cache,
    # )
    # assert logger is not None
    # embodied.run.eval_inference_only(
    #     agent, train_env, eval_env, train_replay, eval_replay, logger, args
    # )


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    argv = normalize_resume_flag(argv)
    parsed, other = Flags(configs=["defaults"]).parse_known(argv)
    event_logger.info("config.cli | parsed={} | remaining={}", parsed, other)
    _get_jax()
    named_configs = _get_agent_module().Agent.configs
    config = Config(named_configs["defaults"])
    for name in parsed.configs:
        cprint(f"config.load | name={name}")
        config = apply_named_config(config, named_configs, name)

    config = Flags(config).parse(other)
    configure_training_speed_flags(config)
    configure_jax_allocator(config)
    config = update_config(config)

    args = create_args(config)
    logdir = Path(args.logdir)
    logger = None
    step = None
    logdir.mkdirs()
    config.save(logdir / "config.yaml")
    output_logs = install_run_output_logs(logdir)
    event_logger.info(
        "run.output_log | stream=stdout | path={}", output_logs.stdout_path
    )
    event_logger.info(
        "run.output_log | stream=stderr | path={}", output_logs.stderr_path
    )

    if args.script not in ["parallel_env", "tune_finetune_policy"]:
        step = Counter()
        logger = make_logger(parsed, logdir, step, config)

    wandb = _get_wandb()
    if wandb.run is not None:
        wandb.config.update(
            {
                "configs": " ".join(parsed.configs),
            },
            allow_val_change=True,
        )

    cleanup = []
    cprint(
        f"run.start | script={args.script} | encoder={config.encoder_type} | "
        f"decoder={config.decoder_type}"
    )

    try:
        if args.script == "train":
            run_train(logger, config, args, logdir, step, cleanup)

        elif args.script == "finetune_policy_seen_traj":
            run_finetune_policy_seen_traj(logger, config, args, logdir, cleanup)

        elif args.script == "profiler":
            # train_env = wrapped_env(config, batch=False)
            # train_env(train_env)
            pass

        elif args.script == "parallel":
            run_parallel(logger, config, args, logdir)

        elif args.script in ["parallel_train_eval", "parallel_train_eval_test"]:
            run_parallel_train_evaluation(logger, config, args, logdir)

        elif args.script == "finetune_wm":
            run_finetune_wm(logger, config, args, logdir, cleanup)

        elif args.script == "finetune_policy":
            run_compute_on_fixed_wm(logger, config, args, logdir, cleanup)

        elif args.script == "tune_finetune_policy":
            tune_finetune_policy(logger, config, logdir, cleanup)

        elif args.script == "create_offline_wm_data":
            create_offline_wm_data(logger, config, logdir, cleanup)

        elif config.run.script == "train_offline_wm":
            # replay_dir = pathlib.Path(args.logdir).parent / args.resume_replay_from  # type: ignore
            run_train_offline_wm(logger, config, args.logdir, cleanup=cleanup)

        elif config.run.script == "eval_offline_wm":
            run_eval_offline_wm(logger, config, cleanup)

        elif args.script == "train_save":
            raise NotImplementedError("train_save")

        elif args.script == "train_eval":
            raise NotImplementedError("train_eval")

        elif args.script == "train_eval_train":
            raise NotImplementedError("train_eval_train")

        elif args.script == "train_holdout":
            raise NotImplementedError("train_holdout")

        elif args.script == "eval_only":
            run_eval_only()

        elif args.script == "parallel_eval":
            run_parallel_eval(logger, config, args)

        elif args.script == "parallel_agent":
            raise NotImplementedError("parallel_agent")

        elif args.script == "parallel_env":
            raise NotImplementedError("parallel_env")

        elif config.run.script == "offline-text":
            raise NotImplementedError("offline-text")

        else:
            raise NotImplementedError(args.script)
    except BaseException:
        event_logger.exception("run.failed | script={}", args.script)
        raise
    finally:
        for obj in cleanup:
            obj.close()
        complete_logging()
        output_logs.close()
        wandb = _get_wandb()
        if wandb.run is not None:
            try:
                save_run_log_artifact(wandb, output_logs.paths)
            except Exception as exc:
                event_logger.warning("run.log_artifact_error | error={}", exc)
        complete_logging()


def make_logger(parsed, logdir, step: "Counter", config):
    multiplier = config.env.get(config.task.split("_")[0], {}).get("repeat", 1)

    outputs = [
        JSONLOutput(logdir, "metrics.jsonl"),
        JSONLOutput(logdir, "scores.jsonl", "(episode/score|real_step)"),
    ]
    # if config.run.script != "finetune_policy":
    #     outputs.append(TerminalOutput(config.filter))

    if config.use_wandb:
        from ledwm.embodied.core.WandBOutput import WandBOutput, create_wandb_init

        run = create_wandb_init(config, logdir)
        outputs.append(
            WandBOutput(
                run,
                config,
                config.filter,
                config.table_keys if config.run.use_table else None,
                config.real_step_log,
                table_names=config.table_names,
            )
        )
        print(f"logger.table | columns={config.table_keys}")

    logger = Logger(step, outputs, multiplier)
    return logger


def wrapped_env(config, batch, **overrides):
    ctor = bind(make_env, config, **overrides)
    if batch and config.envs.parallel != "none":
        ctor = bind(Parallel, ctor, config.envs.parallel)
    if config.envs.restart:
        ctor = bind(wrappers.RestartOnException, ctor)
    if batch:
        envs = [ctor() for _ in range(config.envs.amount)]
        return BatchEnv(envs, (config.envs.parallel != "none"))
    else:
        return ctor()


def make_env(config, **overrides):
    from ledwm.embodied.envs import from_gym

    suite, task = config.task.split("_", 1)
    # suite = "messenger" if "lwm" in suite else suite

    ctor = {
        "dummy": "embodied.envs.dummy:Dummy",
        "gym": "embodied.envs.from_gym:FromGym",
        "dm": "embodied.envs.from_dmenv:FromDM",
        "crafter": "embodied.envs.crafter:Crafter",
        "dmc": "embodied.envs.dmc:DMC",
        "atari": "embodied.envs.atari:Atari",
        "atari100k": "embodied.envs.atari:Atari",
        "dmlab": "embodied.envs.dmlab:DMLab",
        "minecraft": "embodied.envs.minecraft:Minecraft",
        "loconav": "embodied.envs.loconav:LocoNav",
        "pinpad": "embodied.envs.pinpad:PinPad",
        "homegrid": "embodied.envs.homegrid:HomeGrid",
        "vln": "embodied.envs.vln:VLNEnv",
        "langroom": "langroom:LangRoom",
        "procgen": lambda task, **kw: from_gym.FromGym(
            f"procgen:procgen-{task}-v0", **kw
        ),
        "messenger": (
            "embodied.envs.MessengerSent:MessengerSent"
            if config.encoder_type == "sent"
            else "embodied.envs.MessengerToken:MessengerToken"
        ),
        "lwm": ("embodied.envs.LWMSent:LWMSent"),
    }[suite]

    if isinstance(ctor, str):
        module, cls = ctor.split(":")
        module = importlib.import_module(module)
        ctor = getattr(module, cls)

    kwargs = config.env.get(suite, {})
    kwargs.update(overrides)
    env = ctor(task, **kwargs)
    return wrap_env(env, config)


def wrap_env(env, config):
    from ledwm.embodied.core import wrappers

    wrapper_config = config.wrapper
    # Env specific wrappers
    if hasattr(env, "wrappers"):
        for w in env.wrappers:
            env = w(env)

    for name, space in env.act_space.items():
        if name == "reset":
            continue
        elif space.discrete:
            env = wrappers.OneHotAction(env, name)
        elif wrapper_config.discretize:
            env = wrappers.DiscretizeAction(env, name, wrapper_config.discretize)
        else:
            env = wrappers.NormalizeAction(env, name)

    env = wrappers.ExpandScalars(env)
    if wrapper_config.length:
        env = wrappers.TimeLimit(
            env,
            wrapper_config.length,
            wrapper_config.reset,
            # wrapper_config.timeout_reward,
        )
    if wrapper_config.checks:
        env = wrappers.CheckSpaces(env)
    for name, space in env.act_space.items():
        if not space.discrete:
            env = wrappers.ClipAction(env, name)
    return env


if __name__ == "__main__":
    main()
