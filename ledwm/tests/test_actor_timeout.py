import types

import pytest

from ledwm.embodied.core.distr import resolve_actor_timeout


def test_actor_timeout_defaults_to_thirty_seconds():
    assert resolve_actor_timeout(types.SimpleNamespace()) == 30


def test_actor_timeout_uses_configured_value():
    args = types.SimpleNamespace(actor_timeout=45)
    assert resolve_actor_timeout(args) == 45


@pytest.mark.parametrize("value", [0, -1])
def test_actor_timeout_must_be_positive(value):
    args = types.SimpleNamespace(actor_timeout=value)
    with pytest.raises(ValueError, match="actor_timeout must be positive"):
        resolve_actor_timeout(args)
