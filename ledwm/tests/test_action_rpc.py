import numpy as np

from ledwm.embodied.run.action_rpc import (
    ACTION_INDEX_KEY,
    compact_action_for_rpc,
    expand_action_from_rpc,
)


class Space:

    def __init__(self, shape, dtype=np.float32, discrete=True):
        self.shape = shape
        self.dtype = dtype
        self.discrete = discrete


def test_compact_onehot_action_batch_to_indices():
    action = np.array([[0, 1, 0], [1, 0, 0]], dtype=np.float32)
    act = {"action": action, "reset": np.array([False, True])}

    compact = compact_action_for_rpc(act, Space((3,)))

    assert "action" not in compact
    np.testing.assert_array_equal(compact[ACTION_INDEX_KEY], np.array([1, 0], dtype=np.uint8))
    np.testing.assert_array_equal(compact["reset"], act["reset"])


def test_expand_action_index_to_onehot_action():
    act = {ACTION_INDEX_KEY: np.uint8(2), "reset": False}

    expanded = expand_action_from_rpc(act, Space((3,)))

    np.testing.assert_array_equal(expanded["action"], np.array([0, 0, 1], dtype=np.float32))
    assert expanded["reset"] is False
    assert ACTION_INDEX_KEY not in expanded


def test_non_discrete_action_is_left_unchanged():
    act = {"action": np.array([0.1, -0.2], dtype=np.float32)}

    compact = compact_action_for_rpc(act, Space((2,), discrete=False))

    assert compact is act
