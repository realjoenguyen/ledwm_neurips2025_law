import os
from types import SimpleNamespace


def test_early_jax_setup_applies_default_platform_allocator(monkeypatch):
    from ledwm import train

    monkeypatch.delenv("XLA_PYTHON_CLIENT_ALLOCATOR", raising=False)
    monkeypatch.delenv("XLA_PYTHON_CLIENT_PREALLOCATE", raising=False)
    monkeypatch.delenv("XLA_FLAGS", raising=False)
    monkeypatch.delenv("TF_CPP_MIN_LOG_LEVEL", raising=False)

    train._early_configure_jax_allocator([])

    assert os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] == "platform"
    assert os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] == "false"
    assert "--xla_gpu_autotune_level=4" in os.environ["XLA_FLAGS"]
    assert "--xla_gpu_enable_triton_gemm=false" not in os.environ["XLA_FLAGS"]
    assert os.environ["TF_CPP_MIN_LOG_LEVEL"] == "2"


def test_early_jax_setup_preserves_existing_non_platform_allocator(monkeypatch):
    from ledwm import train

    monkeypatch.setenv("XLA_PYTHON_CLIENT_ALLOCATOR", "bfc")
    monkeypatch.delenv("XLA_FLAGS", raising=False)
    monkeypatch.delenv("TF_CPP_MIN_LOG_LEVEL", raising=False)

    train._early_configure_jax_allocator([])

    assert os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] == "bfc"
    assert "XLA_FLAGS" not in os.environ
    assert "TF_CPP_MIN_LOG_LEVEL" not in os.environ


def test_early_jax_setup_preserves_existing_xla_autotune_flag(monkeypatch):
    from ledwm import train

    monkeypatch.delenv("XLA_PYTHON_CLIENT_ALLOCATOR", raising=False)
    monkeypatch.delenv("XLA_PYTHON_CLIENT_PREALLOCATE", raising=False)
    monkeypatch.setenv("XLA_FLAGS", "--xla_gpu_autotune_level=-1")
    monkeypatch.delenv("TF_CPP_MIN_LOG_LEVEL", raising=False)

    train._early_configure_jax_allocator([])

    assert os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] == "platform"
    assert os.environ["XLA_FLAGS"] == "--xla_gpu_autotune_level=-1"


def test_jax_device_selection_allows_dedicated_policy_device():
    from ledwm import train

    config = SimpleNamespace(
        jax=SimpleNamespace(policy_devices=(0,), train_devices=(1, 2, 3))
    )

    train._validate_jax_device_selection(config, [object()] * 4)


def test_jax_device_selection_rejects_out_of_range_device():
    from ledwm import train

    config = SimpleNamespace(
        jax=SimpleNamespace(policy_devices=(0,), train_devices=(1, 2, 4))
    )

    try:
        train._validate_jax_device_selection(config, [object()] * 4)
    except AssertionError as exc:
        assert "visible JAX device indices" in str(exc)
    else:
        raise AssertionError("expected out-of-range train device to fail")


def test_train_batch_size_floors_to_train_device_multiple():
    from ledwm import train

    assert train._resolve_train_batch_size(500, (1, 2, 3)) == 498


def test_train_batch_size_keeps_already_divisible_value():
    from ledwm import train

    assert train._resolve_train_batch_size(498, (1, 2, 3)) == 498


def test_train_batch_size_floor_rejects_too_small_value():
    from ledwm import train

    try:
        train._resolve_train_batch_size(2, (1, 2, 3))
    except AssertionError as exc:
        assert "at least" in str(exc)
    else:
        raise AssertionError("expected too-small batch size to fail")
