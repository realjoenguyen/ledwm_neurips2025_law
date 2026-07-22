import fcntl
import hashlib
import os
import pathlib
import pickle
import tempfile
from dataclasses import dataclass

import numpy as np


_CACHE_ENV = "LEDWM_SENTENCE_EMBED_CACHE_DIR"


@dataclass(frozen=True)
class SentenceEmbeddingCacheInfo:
    array_path: pathlib.Path | None
    created: bool
    memory_mapped: bool


def _read_source(path, t5_sent):
    with path.open("rb") as handle:
        sent2id, id2sentemb = pickle.load(handle)
    if not t5_sent:
        sent2id, id2sentemb = id2sentemb, sent2id
    return sent2id, np.asarray(id2sentemb)


def _cache_root(cache_dir=None):
    configured = os.environ.get(_CACHE_ENV) if cache_dir is None else cache_dir
    if configured is not None and str(configured).lower() in (
        "",
        "0",
        "false",
        "off",
    ):
        return None
    if configured:
        return pathlib.Path(configured)
    return pathlib.Path(tempfile.gettempdir()) / (
        f"ledwm-sentence-embeddings-{os.getuid()}"
    )


def _cache_key(path, t5_sent):
    stat = path.stat()
    identity = (
        f"{os.path.abspath(path)}\0{stat.st_size}\0{stat.st_mtime_ns}\0"
        f"{int(t5_sent)}"
    )
    return hashlib.sha256(identity.encode()).hexdigest()


def _write_cache(source, t5_sent, metadata_path, array_path):
    sent2id, id2sentemb = _read_source(source, t5_sent)
    suffix = f".tmp-{os.getpid()}"
    metadata_tmp = metadata_path.with_name(metadata_path.name + suffix)
    array_tmp = array_path.with_name(array_path.name + suffix)
    try:
        with array_tmp.open("wb") as handle:
            np.save(handle, id2sentemb, allow_pickle=False)
        with metadata_tmp.open("wb") as handle:
            pickle.dump(sent2id, handle, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(array_tmp, array_path)
        os.replace(metadata_tmp, metadata_path)
    finally:
        for path in (metadata_tmp, array_tmp):
            try:
                path.unlink()
            except FileNotFoundError:
                pass


def load_sentence_embeddings(source, t5_sent=False, cache_dir=None):
    """Load a pickle-backed embedding table through a shared read-only mmap.

    Spawned environment workers cannot inherit the trainer's Python objects.
    The first process converts the source pickle into a node-local ``.npy``;
    all later workers map the same pages instead of deserializing and privately
    allocating the array again. Set ``LEDWM_SENTENCE_EMBED_CACHE_DIR=off`` to
    use the original direct-pickle path.
    """
    source = pathlib.Path(source)
    root = _cache_root(cache_dir)
    if root is None:
        sent2id, id2sentemb = _read_source(source, t5_sent)
        return sent2id, id2sentemb, SentenceEmbeddingCacheInfo(None, False, False)
    try:
        root.mkdir(mode=0o700, parents=True, exist_ok=True)
        key = _cache_key(source, t5_sent)
        metadata_path = root / f"{key}.sent2id.pkl"
        array_path = root / f"{key}.embeddings.npy"
        lock_path = root / f"{key}.lock"
        created = False
        with lock_path.open("a+b") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            try:
                if not metadata_path.is_file() or not array_path.is_file():
                    raise FileNotFoundError
                with metadata_path.open("rb") as handle:
                    sent2id = pickle.load(handle)
                id2sentemb = np.load(array_path, mmap_mode="r", allow_pickle=False)
            except (OSError, ValueError, EOFError, pickle.PickleError):
                for path in (metadata_path, array_path):
                    try:
                        path.unlink()
                    except FileNotFoundError:
                        pass
                _write_cache(source, t5_sent, metadata_path, array_path)
                created = True
                with metadata_path.open("rb") as handle:
                    sent2id = pickle.load(handle)
                id2sentemb = np.load(array_path, mmap_mode="r", allow_pickle=False)
        return sent2id, id2sentemb, SentenceEmbeddingCacheInfo(
            array_path, created, True
        )
    except (OSError, ValueError, EOFError, pickle.PickleError):
        # A cache directory can be full, read-only, or left incomplete after a
        # node failure. The optimization must never prevent the env from starting.
        sent2id, id2sentemb = _read_source(source, t5_sent)
        return sent2id, id2sentemb, SentenceEmbeddingCacheInfo(None, False, False)


def should_log_sentence_embeddings(cache):
    configured = os.environ.get("LEDWM_SENTENCE_EMBED_LOG_EACH", "0").lower()
    return (
        cache.created
        or not cache.memory_mapped
        or configured in ("1", "true", "yes")
    )
