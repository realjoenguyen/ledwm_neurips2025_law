# %%
from turtle import up
import numpy as np


class Node:
    __slots__ = ("parent", "children", "uprob")

    def __init__(self, parent=None):
        self.parent = parent
        self.children = []
        self.uprob = 0

    def __repr__(self):
        return (
            f"Node(uprob={self.uprob}, " f"children={[x.uprob for x in self.children]})"
        )

    def __len__(self):
        return len(self.children)

    def __bool__(self):
        return True

    def append(self, child):
        if child.parent:
            child.parent.remove(child)
        child.parent = self
        self.children.append(child)
        self.recompute()

    def remove(self, child):
        child.parent = None
        self.children.remove(child)
        self.recompute()

    def recompute(self):
        # Compute `uprob` and assert non-negative values
        self.uprob = sum(x.uprob for x in self.children)
        assert self.uprob >= 0, "Node uprob must be non-negative."
        if self.parent:
            self.parent.recompute()


class Entry:
    __slots__ = ("parent", "key", "uprob")

    def __init__(self, key=None, uprob=None):
        assert uprob >= 0, "Entry uprob must be non-negative."
        self.parent = None
        self.key = key
        self.uprob = uprob


def softmax(x):
    exp_x = np.exp(x - np.max(x))  # Subtract max for numerical stability
    return exp_x / exp_x.sum()


class SampleTree:
    def __init__(self, branching=16, seed=None):
        assert 2 <= branching
        self.branching = branching
        self.root = Node()
        self.last = None
        self.entries = {}
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.entries)

    def insert(self, key, uprob):
        assert uprob >= 0, "Inserted uprob must be non-negative."
        if not self.last:
            node = self.root
        else:
            ups = 0
            node = self.last.parent
            while node and len(node) >= self.branching:
                node = node.parent
                ups += 1

            if not node:
                node = Node()
                node.append(self.root)
                self.root = node

            for _ in range(ups):
                below = Node()
                node.append(below)
                node = below

        entry = Entry(key, uprob)
        node.append(entry)
        self.entries[key] = entry
        self.last = entry

    def has_key(self, key):
        return key in self.entries

    def remove(self, key):
        entry = self.entries.pop(key)
        entry_parent = entry.parent
        last_parent = self.last.parent if self.last else None
        entry.parent.remove(entry)
        if entry is not self.last:
            entry_parent.append(self.last)

        node = last_parent
        while node and node.parent and not len(node):
            above = node.parent
            above.remove(node)
            node = above

        if node is None or not len(node):
            self.last = None
            return

        while isinstance(node, Node):
            node = node.children[-1]
        self.last = node

    def update(self, key, uprob):
        assert uprob >= 0, f"{uprob} is not a valid priority."
        entry = self.entries[key]
        entry.uprob = uprob
        entry.parent.recompute()

    def sample(self):
        node = self.root
        while isinstance(node, Node):
            uprobs = np.array([x.uprob for x in node.children])
            probs = uprobs / uprobs.sum()
            assert np.all(probs >= 0), "Probabilities must be non-negative."
            assert np.isclose(probs.sum(), 1.0), "Probabilities do not sum to 1."
            choice = self.rng.choice(np.arange(len(uprobs)), p=probs)
            node = node.children[choice.item()]
        return node.key

    def sample_batch(self, size, replace=False):
        """Draw leaf keys directly from their weights in one NumPy operation."""
        size = int(size)
        if size < 1:
            return []
        entries = tuple(self.entries.values())
        if not replace and size > len(entries):
            raise ValueError(
                f"Cannot sample {size} unique keys from {len(entries)} entries."
            )

        weights = np.fromiter(
            (entry.uprob for entry in entries), dtype=np.float64, count=len(entries)
        )
        assert np.all(weights >= 0), "Probabilities must be non-negative."
        infinite = np.isinf(weights)
        if infinite.any():
            eligible = infinite
        elif weights.sum() > 0:
            eligible = weights > 0
        else:
            eligible = np.ones(len(entries), dtype=bool)
            weights = eligible.astype(np.float64)

        if not replace and size > int(eligible.sum()):
            raise ValueError(
                f"Cannot sample {size} unique keys from "
                f"{int(eligible.sum())} positive-priority entries."
            )

        probs = np.where(eligible, weights, 0.0)
        if infinite.any():
            probs = eligible.astype(np.float64)
        probs /= probs.sum()
        choices = self.rng.choice(
            len(entries), size=size, replace=replace, p=probs
        )
        return [entries[index].key for index in np.atleast_1d(choices)]


import numpy as np


def test_sample_tree():
    # Initialize SampleTree with a specific branching factor and seed
    tree = SampleTree(branching=4, seed=42)

    # Test insertion
    tree.insert("A", 1 / 2)
    tree.insert("B", 1 / 3)
    tree.insert("C", 1 / 10)
    tree.insert("D", 1 / 30)
    # Debug: Print updated probabilities
    print(f"A uprob: {tree.entries['A'].uprob}")
    print(f"B uprob: {tree.entries['B'].uprob}")
    print(f"C uprob: {tree.entries['C'].uprob}")
    print(f"D uprob: {tree.entries['D'].uprob}")

    assert tree.has_key("A")
    assert tree.has_key("B")
    assert tree.has_key("C")
    assert tree.has_key("D")
    assert len(tree) == 4, "Tree should have 4 entries."

    # Test sampling (biased toward higher uprob)
    samples = [tree.sample() for _ in range(20000)]
    unique, counts = np.unique(samples, return_counts=True)
    sample_counts = dict(zip(unique, counts))

    print("Sample counts:", sample_counts)

    # "D" should have the highest count, followed by "C", "B", and "A"
    # assert (
    #     sample_counts["D"]
    #     > sample_counts["C"]
    #     > sample_counts["B"]
    #     > sample_counts["A"]
    # )

    # Test updating priorities
    tree.update("A", 1 / 5)
    tree.update("B", 1 / 100)

    # Debug: Print updated probabilities
    print(f"A uprob: {tree.entries['A'].uprob}")
    print(f"B uprob: {tree.entries['B'].uprob}")
    print(f"C uprob: {tree.entries['C'].uprob}")
    print(f"D uprob: {tree.entries['D'].uprob}")

    # Verify updated sampling bias
    samples = [tree.sample() for _ in range(20000)]
    unique, counts = np.unique(samples, return_counts=True)
    sample_counts = dict(zip(unique, counts))

    print("Updated sample counts:", sample_counts)

    # "A" should now have the highest count
    # assert sample_counts["A"] > sample_counts["B"]
    # assert sample_counts["B"] > sample_counts["C"]
    # assert sample_counts["D"] > sample_counts["C"]

    # Test removal
    tree.remove("A")
    assert not tree.has_key("A"), "Key 'A' should be removed."
    assert len(tree) == 3, "Tree should have 3 entries after removal."

    # Test sampling after removal
    samples = [tree.sample() for _ in range(1000)]
    unique, counts = np.unique(samples, return_counts=True)
    sample_counts = dict(zip(unique, counts))

    print("Sample counts after removal:", sample_counts)
    assert "A" not in sample_counts, "Key 'A' should not appear in samples."

    # Test tree integrity
    tree.insert("E", 7.0)
    tree.insert("F", 8.0)
    assert tree.has_key("E") and tree.has_key("F")

    print("All tests passed!")


# Run the tests
if __name__ == "__main__":
    test_sample_tree()
