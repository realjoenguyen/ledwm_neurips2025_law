import json
from pathlib import Path

from termcolor import cprint
from ledwm.embodied.replay import generic, limiters, selectors
# ReplayEps (scipy) imported lazily in make_replay when needed

MAX_BS = 200
MAX_BATCH_LENGTH = 32


class Uniform(generic.GenericReplay):
    def __init__(
        self,
        length,  # batch_length
        is_eval,
        capacity=None,  # config.replay.size
        directory=None,
        chunks=1024,
        min_size=1,
        samples_per_insert=None,
        tolerance=1e4,
        seed=0,
        load_directories=None,
        dataset_zero_keys=None,
        overfit_eps=False,
        train_ratio=None,
        online=False,
        config=None,
    ):
        if samples_per_insert:
            limiter = limiters.SamplesPerInsert(
                samples_per_insert, tolerance, min_size, capacity, overfit_eps
            )
        else:
            limiter = limiters.MinSize(min_size)
        assert not capacity or min_size <= capacity, (min_size, capacity)
        super().__init__(
            batch_length=length,
            capacity=capacity,
            remover=selectors.Fifo(),
            sampler=selectors.Uniform(seed),
            limiter=limiter,
            directory=directory,
            chunks=chunks,
            load_directories=load_directories,
            dataset_zero_keys=dataset_zero_keys,
            train_ratio=train_ratio,
            is_eval=is_eval,
            config=config,
            min_size=min_size,
            online=online,
        )


class Queue(generic.GenericReplay):
    pass


EVAL_RATIO = 256
PREFILL_MANIFEST = "replay_manifest.json"


def _replay_prefill_signature(config, batch_length):
    """Fields that must match before raw steps are reused by another run."""
    return {
        "version": 1,
        "task": str(config.task),
        "batch_length": int(batch_length),
        "replay_type": str(config.replay.type),
        "is_first": bool(config.replay.is_first),
        "imbalance_reward": str(config.replay.imbalance_reward),
        "rew_smooth_mode": str(config.rew_smooth_mode),
        "rew_smooth_amt": float(config.rew_smooth_amt),
        "dataset_exclude_keys": sorted(config.dataset_exclude_keys or []),
    }


def _write_replay_manifest(directory, signature):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / PREFILL_MANIFEST
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(signature, sort_keys=True, indent=2) + "\n")
    temporary.replace(path)


def _latest_compatible_prefill(directory, signature):
    """Return the newest sibling run with saved, compatible replay chunks."""
    current = Path(directory).resolve()
    run_root = current.parent.parent
    candidates = []
    for candidate in run_root.glob("*/episodes"):
        try:
            candidate = candidate.resolve()
            if candidate == current or not any(candidate.glob("*.npz")):
                continue
            manifest = candidate / PREFILL_MANIFEST
            if not manifest.exists() or json.loads(manifest.read_text()) != signature:
                continue
            candidates.append(candidate)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _resolve_replay_prefill(config, directory, signature):
    if directory is None or config.replay.resume:
        return None
    value = config.replay.flat.get("prefill", "")
    if value in (None, "", False, "false", "False", "none", "None"):
        return None
    if str(value).lower() == "auto":
        source = _latest_compatible_prefill(directory, signature)
        if source is None:
            cprint("replay.prefill | source=none | reason=no_compatible_chunks", "yellow")
            return None
        cprint(f"replay.prefill | source={source} | mode=auto", "green")
        return [source]
    sources = value if isinstance(value, (list, tuple)) else [value]
    sources = [Path(source).expanduser() for source in sources if str(source)]
    cprint(f"replay.prefill | source={sources} | mode=explicit", "green")
    return sources or None


def _resolve_replay_min_size(config, batch_size, is_eval):
    if is_eval:
        return int(batch_size)
    replay_min_size = 0
    try:
        replay_min_size = int(config.replay.flat.get("min_size", 0))
    except AttributeError:
        replay_min_size = 0
    replay_min_size = int(batch_size) if replay_min_size <= 0 else replay_min_size
    # Prioritized S1 batches are sampled without replacement. Starting below one
    # complete batch would either block inside sampling or violate uniqueness.
    if config.replay.type == "prioritize":
        replay_min_size = max(replay_min_size, int(batch_size))
    return replay_min_size


def make_replay(
    config,
    directory=None,
    is_eval=False,  # reduce replay size for eval // 10
    rate_limit=False,  # True if parallel_train
    load_directories=None,
    batch_length=None,
    batch_size=None,
    smooth_reward=True,  # False: for finetune_policy
    online=False,
    size=None,
    **kwargs,
):
    mode = "eval" if is_eval else "train"
    cprint(f"replay.create | mode={mode} | directory={directory}")
    batch_length = config.batch_length if batch_length is None else batch_length
    batch_size = config.batch_size if batch_size is None else batch_size
    cprint(
        f"replay.config | mode={mode} | batch_length={batch_length} | "
        f"batch_size={batch_size}",
        "yellow",
    )
    min_size = _resolve_replay_min_size(config, batch_size, is_eval)
    if min_size != batch_size:
        cprint(f"replay.config | mode={mode} | min_size={min_size}", "yellow")
    if size is None:
        size = (
            config.replay.size // 2
            if is_eval and not config.jax.debug
            else config.replay.size
        )

    if (
        config.run.from_checkpoint != ""
        and config.replay.resume
        and "offline_wm" not in config.run.script
    ):
        directory = Path(config.run.from_checkpoint).parent / "episodes"
        cprint(f"replay.resume | directory={directory}", "green")

    signature = None
    if not is_eval and directory is not None:
        signature = _replay_prefill_signature(config, batch_length)
        auto_load_directories = _resolve_replay_prefill(config, directory, signature)
        if auto_load_directories:
            load_directories = list(load_directories or []) + auto_load_directories

    if not (load_directories and load_directories[0]):
        load_directories = None

    if config.replay.type == "uniform" or is_eval:
        kw = {"load_directories": load_directories}
        if (
            rate_limit
            and config.run.train_ratio > 0
            and "finetune_wm" not in config.run.script
        ):
            kw["samples_per_insert"] = (
                EVAL_RATIO / batch_length
                if is_eval
                else config.run.train_ratio / batch_length
            )
            kw["tolerance"] = (
                MAX_BATCH_LENGTH * MAX_BS * config.replay.tolerance_rate
                if not (config.run.overfit_eps or config.jax.debug)
                else batch_size
            )

        kw["train_ratio"] = EVAL_RATIO if is_eval else config.run.train_ratio
        cprint(f"replay.backend | type=uniform | options={kw}", "yellow")
        replay = Uniform(
            batch_length,
            is_eval,
            size,
            directory,
            config=config,
            min_size=min_size,
            online=online,
            **kw,
        )

    elif config.replay.type == "reverb":
        raise NotImplementedError("Reverb")

    elif config.replay.type == "chunks":
        raise NotImplementedError("Chunks")

    elif config.replay.type == "curious":
        raise NotImplementedError("CuriousReplay")

    elif config.replay.type == "prioritize":
        from ledwm.embodied.replay.Prioritized import PrioritizedReplay

        assert not is_eval, "PrioritizedReplay not supported for eval"
        kw = {"load_directories": load_directories}

        if rate_limit and config.run.train_ratio > 0:
            kw["samples_per_insert"] = config.run.train_ratio / batch_length
            kw["tolerance"] = (
                config.replay.tolerance_rate * MAX_BS * MAX_BATCH_LENGTH
                if not (config.run.overfit_eps or config.jax.debug)
                else batch_size
            )

        cprint(f"replay.backend | type=prioritized | options={kw}")
        replay = PrioritizedReplay(
            batch_length,
            config.run.train_ratio,
            size,
            directory,
            config=config,
            is_eval=is_eval,
            min_size=min_size,
            **kw,
            **config.replay.prioritize_hyper,
        )

    else:
        raise NotImplementedError(config.replay)

    if config.rew_smooth_mode == "gaussian":
        assert config.rew_smooth_amt > 0, config.rew_smooth_amt
    # else:
    #     assert config.rew_smooth_amt == 0, config.rew_smooth_amt

    if config.run.script != "finetune_policy":
        from ledwm.embodied.run.smoothing import ReplayEps

        replay = ReplayEps(
            replay,
            sigma=config.rew_smooth_amt if config.rew_smooth_mode == "gaussian" else 0,
            config=config,
        )

    if signature is not None:
        _write_replay_manifest(directory, signature)

    return replay
