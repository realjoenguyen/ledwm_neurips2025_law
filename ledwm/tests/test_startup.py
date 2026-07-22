import os
import importlib
import sys
import types


def load_startup():
    try:
        return importlib.import_module("ledwm.startup")
    except ModuleNotFoundError:
        return None


def test_jax_cache_announcement_is_once_per_process_tree(monkeypatch):
    startup = load_startup()
    assert startup is not None, "ledwm.startup module is required"
    monkeypatch.delenv(startup.JAX_CACHE_ANNOUNCED_ENV, raising=False)

    assert startup.should_announce_jax_cache() is True
    assert os.environ[startup.JAX_CACHE_ANNOUNCED_ENV] == "1"
    assert startup.should_announce_jax_cache() is False


def test_jax_cache_announcement_honors_inherited_env(monkeypatch):
    startup = load_startup()
    assert startup is not None, "ledwm.startup module is required"
    monkeypatch.setenv(startup.JAX_CACHE_ANNOUNCED_ENV, "1")

    assert startup.should_announce_jax_cache() is False


def test_tensorflow_cpp_warning_filter_defaults_to_warning_suppression(monkeypatch):
    startup = load_startup()
    assert startup is not None, "ledwm.startup module is required"
    monkeypatch.delenv("TF_CPP_MIN_LOG_LEVEL", raising=False)
    monkeypatch.delenv("TF_ENABLE_ONEDNN_OPTS", raising=False)

    startup.configure_tensorflow_cpp_warnings()

    assert os.environ["TF_CPP_MIN_LOG_LEVEL"] == "2"
    assert os.environ["TF_ENABLE_ONEDNN_OPTS"] == "0"


def test_tensorflow_cpp_warning_filter_preserves_user_setting(monkeypatch):
    startup = load_startup()
    assert startup is not None, "ledwm.startup module is required"
    monkeypatch.setenv("TF_CPP_MIN_LOG_LEVEL", "0")
    monkeypatch.setenv("TF_ENABLE_ONEDNN_OPTS", "1")

    startup.configure_tensorflow_cpp_warnings()

    assert os.environ["TF_CPP_MIN_LOG_LEVEL"] == "0"
    assert os.environ["TF_ENABLE_ONEDNN_OPTS"] == "1"


def test_numeric_threading_defaults_to_one(monkeypatch):
    startup = load_startup()
    assert startup is not None, "ledwm.startup module is required"
    for name in startup.NUMERIC_THREAD_ENV_VARS:
        monkeypatch.delenv(name, raising=False)

    startup.configure_numeric_threading()

    for name in startup.NUMERIC_THREAD_ENV_VARS:
        assert os.environ[name] == "1"


def test_numeric_threading_preserves_user_setting(monkeypatch):
    startup = load_startup()
    assert startup is not None, "ledwm.startup module is required"
    for name in startup.NUMERIC_THREAD_ENV_VARS:
        monkeypatch.setenv(name, "4")

    startup.configure_numeric_threading()

    for name in startup.NUMERIC_THREAD_ENV_VARS:
        assert os.environ[name] == "4"


def test_jax_cache_ir_normalizes_only_process_specific_function_addresses():
    startup = load_startup()
    first = (
        b"reducer_fn at 0x1553f5291940> consts=() "
        b"literal=0x1234 path=/tmp/0x5678"
    )
    second = (
        b"reducer_fn at 0x1553e6867940> consts=() "
        b"literal=0x1234 path=/tmp/0x5678"
    )

    normalized_first = startup.normalize_jax_cache_ir(first)
    normalized_second = startup.normalize_jax_cache_ir(second)

    assert normalized_first == normalized_second
    assert b"literal=0x1234 path=/tmp/0x5678" in normalized_first
    assert len(normalized_first) == len(first)


def test_jax_cache_key_patch_skips_newer_private_signature():
    startup = load_startup()

    def modern_hash_computation(hash_obj, module, ignore_callbacks):
        return None

    cache_key = types.SimpleNamespace(
        _hash_computation=modern_hash_computation,
    )

    assert startup._stabilize_jax_cache_key(cache_key) is False
    assert cache_key._hash_computation is modern_hash_computation


def test_jax_compilation_cache_falls_back_to_experimental_api(tmp_path, monkeypatch):
    startup = load_startup()
    assert startup is not None, "ledwm.startup module is required"
    monkeypatch.setenv("JAX_COMPILATION_CACHE_DIR", str(tmp_path))
    calls = []

    class Config:
        def update(self, name, value):
            if name == "jax_compilation_cache_dir":
                raise AttributeError("Unrecognized config option")
            calls.append(("config", name, value))

    cache = types.SimpleNamespace(
        initialize_cache=lambda path: calls.append(("initialize_cache", path))
    )
    jax = types.SimpleNamespace(config=Config())

    enabled = startup.configure_jax_compilation_cache(jax, cache)

    assert enabled is True
    assert ("initialize_cache", str(tmp_path)) in calls
    assert ("config", "jax_persistent_cache_min_compile_time_secs", 0) in calls


def test_jax_compilation_cache_suppresses_experimental_api_stdout(
    tmp_path, monkeypatch, capsys
):
    startup = load_startup()
    assert startup is not None, "ledwm.startup module is required"
    monkeypatch.setenv("JAX_COMPILATION_CACHE_DIR", str(tmp_path))

    class Config:
        def update(self, name, value):
            if name == "jax_compilation_cache_dir":
                raise AttributeError("Unrecognized config option")

    def initialize_cache(path):
        print(f"Initialized persistent compilation cache at {path}")
        print(f"Initialized persistent compilation cache at {path}", file=sys.stderr)

    cache = types.SimpleNamespace(initialize_cache=initialize_cache)
    jax = types.SimpleNamespace(config=Config())

    assert startup.configure_jax_compilation_cache(jax, cache) is True
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""
