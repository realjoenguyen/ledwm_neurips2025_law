import numpy as np


ACTION_INDEX_KEY = "action_index"


def _index_dtype(count):
    if count <= np.iinfo(np.uint8).max + 1:
        return np.uint8
    if count <= np.iinfo(np.uint16).max + 1:
        return np.uint16
    return np.int32


def _can_compact_action_space(space):
    if space is None:
        return False
    shape = getattr(space, "shape", None)
    dtype = getattr(space, "dtype", None)
    return (
        bool(getattr(space, "discrete", False))
        and isinstance(shape, tuple)
        and len(shape) == 1
        and dtype is not None
        and np.issubdtype(np.dtype(dtype), np.floating)
    )


def compact_action_for_rpc(act, action_space, key="action"):
    if key not in act or not _can_compact_action_space(action_space):
        return act

    action = np.asarray(act[key])
    if action.shape[-1:] != tuple(action_space.shape):
        return act

    compact = {k: v for k, v in act.items() if k != key}
    count = int(action_space.shape[0])
    compact[ACTION_INDEX_KEY] = np.argmax(action, axis=-1).astype(
        _index_dtype(count), copy=False
    )
    return compact


def expand_action_from_rpc(act, action_space, key="action"):
    if key in act or ACTION_INDEX_KEY not in act or not _can_compact_action_space(action_space):
        return act

    index = int(np.asarray(act[ACTION_INDEX_KEY]))
    action = np.zeros(action_space.shape, dtype=action_space.dtype)
    action[index] = 1.0

    expanded = {k: v for k, v in act.items() if k != ACTION_INDEX_KEY}
    expanded[key] = action
    return expanded
