from ledwm.embodied.core.flags import Flags


def test_parse_bool_flags_accept_lowercase_values():
    config = {
        "jax": {
            "profiler": False,
            "prealloc": True,
        }
    }

    parsed = Flags(config).parse(
        ["--jax.profiler", "true", "--jax.prealloc", "false"]
    )

    assert parsed.jax.profiler is True
    assert parsed.jax.prealloc is False


def test_parse_bool_flags_reject_unknown_values():
    config = {"jax": {"profiler": False}}

    try:
        Flags(config).parse(["--jax.profiler", "yes"])
    except TypeError as exc:
        assert "Expected bool but got 'yes'" in str(exc)
    else:
        raise AssertionError("Expected TypeError for invalid bool flag value")
