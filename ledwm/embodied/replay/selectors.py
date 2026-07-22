from collections import OrderedDict

import numpy as np


class Fifo:
    def __init__(self):
        # OrderedDict gives O(1) removal by key without maintaining deque
        # indices, which become stale after popleft().
        self.queue = OrderedDict()

    def __len__(self):
        return len(self.queue)

    def __call__(self):
        try:
            return next(iter(self.queue))
        except StopIteration:
            raise IndexError("FIFO is empty") from None

    # contains
    def __contains__(self, key):
        return key in self.queue

    def __setitem__(self, key, steps):
        self.queue[key] = None

    def __delitem__(self, key):
        del self.queue[key]


# class Fifo:
#     def __init__(self):
#         self.queue = deque()

#     def __call__(self):
#         return self.queue[0]

#     def __setitem__(self, key, steps):
#         self.queue.append(key)

#     def __delitem__(self, key):
#         if self.queue[0] == key:
#             self.queue.popleft()
#         else:
#             # TODO: This is extremely slow.
#             self.queue.remove(key)


class Uniform:
    def __init__(self, seed=0):
        self.indices = {}
        self.keys = []
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.keys)

    def __call__(self, past_sample_ids=None):
        index = self.rng.integers(0, len(self.keys)).item()
        return self.keys[index]

    def __setitem__(self, key, steps):
        self.indices[key] = len(self.keys)
        self.keys.append(key)

    def __contains__(self, key):
        return key in self.indices

    def __delitem__(self, key):
        index = self.indices.pop(key)
        last = self.keys.pop()
        if index != len(self.keys):
            self.keys[index] = last
            self.indices[last] = index
