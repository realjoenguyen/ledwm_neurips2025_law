import contextlib
import inspect
import io
import os
import pathlib
import re


JAX_CACHE_ANNOUNCED_ENV = "LEDWM_JAX_CACHE_ANNOUNCED"
NUMERIC_THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
)
_FUNCTION_ADDRESS_RE = re.compile(rb"(?<= at 0x)[0-9a-fA-F]+(?=>)")


def configure_numeric_threading(default="1"):
    for name in NUMERIC_THREAD_ENV_VARS:
        os.environ.setdefault(name, str(default))


def configure_tensorflow_cpp_warnings():
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")


def should_announce_jax_cache():
    if os.environ.get(JAX_CACHE_ANNOUNCED_ENV) == "1":
        return False
    os.environ[JAX_CACHE_ANNOUNCED_ENV] = "1"
    return True


def _update_jax_config_if_supported(jax_module, name, value):
    try:
        jax_module.config.update(name, value)
    except (AttributeError, ValueError):
        return False
    return True


def normalize_jax_cache_ir(serialized_ir):
    """Remove process-specific function addresses from JAX cache-key IR."""
    return _FUNCTION_ADDRESS_RE.sub(
        lambda match: b"0" * len(match.group(0)), serialized_ir
    )


def _stabilize_jax_cache_key(cache_key_module=None):
    # JAX 0.4.13 leaves Python callable addresses in canonicalized StableHLO,
    # notably in argmin/argmax reducer representations. They are debug identity,
    # not executable semantics, but otherwise make every process use a new key.
    if cache_key_module is None:
        try:
            from jax._src import cache_key as cache_key_module
        except ImportError:
            from jax._src import compilation_cache as cache_key_module

    if getattr(cache_key_module, "_ledwm_stable_function_addresses", False):
        return True

    # JAX 0.4.13 used a two-argument private hook. Newer JAX versions add
    # cache-key inputs (for example callback handling) and already canonicalize
    # newer StableHLO. Do not replace an unknown private API and accidentally
    # discard fields that belong in the cache key.
    if len(inspect.signature(cache_key_module._hash_computation).parameters) != 2:
        return False

    def stable_hash_computation(hash_obj, module):
        if cache_key_module.config.jax_compilation_cache_include_metadata_in_key:
            serialized_ir = cache_key_module._serialize_ir(module)
        else:
            serialized_ir = cache_key_module._canonicalize_ir(module)
        hash_obj.update(normalize_jax_cache_ir(serialized_ir))

    cache_key_module._hash_computation = stable_hash_computation
    cache_key_module._ledwm_stable_function_addresses = True
    return True


def configure_jax_compilation_cache(jax_module, cache_module=None):
    if os.environ.get("JAX_COMPILATION_CACHE_ENABLE", "1").lower() in (
        "0",
        "false",
        "no",
    ):
        return False
    cache_dir = os.environ.get(
        "JAX_COMPILATION_CACHE_DIR",
        os.path.expanduser("~/.cache/ledwm_jax_cache"),
    )
    pathlib.Path(cache_dir).mkdir(parents=True, exist_ok=True)
    use_internal_cache = cache_module is None
    configured = _update_jax_config_if_supported(
        jax_module, "jax_compilation_cache_dir", cache_dir
    )
    if not configured:
        if cache_module is None:
            from jax.experimental.compilation_cache import compilation_cache

            cache_module = compilation_cache
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
            io.StringIO()
        ):
            cache_module.initialize_cache(cache_dir)
        configured = True

    _update_jax_config_if_supported(
        jax_module, "jax_persistent_cache_min_compile_time_secs", 0
    )
    _update_jax_config_if_supported(
        jax_module, "jax_persistent_cache_min_entry_size_bytes", 0
    )
    if use_internal_cache:
        _stabilize_jax_cache_key()
    return configured
