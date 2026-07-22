import numpy as np

from ledwm.embodied.run.timing import has_tree_leaves


def test_has_tree_leaves_rejects_empty_info_tree():
    assert not has_tree_leaves({})
    assert not has_tree_leaves({"nested": {}})


def test_has_tree_leaves_accepts_policy_info_values():
    assert has_tree_leaves({"entropy": np.array([1.0])})
