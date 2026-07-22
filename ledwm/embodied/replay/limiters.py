import threading

import numpy as np
from termcolor import cprint


class MinSize:
    def __init__(self, minimum):
        assert 1 <= minimum, minimum
        self.minimum = minimum
        self.size = 0
        self.lock = threading.Lock()

    def want_load(self):
        with self.lock:
            self.size += 1
        return True, "ok"

    def want_insert(self):
        with self.lock:
            self.size += 1
        return True, "ok"

    def want_remove(self):
        with self.lock:
            if self.size < 1:
                return False, "is empty"
            self.size -= 1
        return True, "ok"

    def want_sample(self):
        if self.size < self.minimum:
            return False, f"too empty: {self.size} < {self.minimum}"
        return True, "ok"


class SamplesPerInsert:
    def __init__(
        self,
        samples_per_insert,
        tolerance,
        min_size=1,
        capacity=None,
        overfit_eps=False,
    ):
        """
        Avoids - constraints:
            - insert too much
            - sample too much
            - sample on empty buffer

        Args:
            samples_per_insert (int): The number of samples to insert per insert operation.
            tolerance (float): The tolerance value used to determine the rate limits.
            min_size (int, optional): The minimum size of the limiter. Defaults to 1.
        """
        assert 1 <= min_size
        self.overfit_eps = overfit_eps
        self.samples_per_insert = samples_per_insert
        self.minimum = min_size
        self.avail = -min_size
        # tolerance = 10 * batch_size -> max_avail = 10 * batch_size * train_ratio / batch_length
        self.tolerance = tolerance
        self.min_avail = -tolerance
        self.max_avail = tolerance * samples_per_insert
        self.size = 0
        self.lock = threading.Lock()
        self.capacity = capacity

    def set_sampler_per_insert(self, samples_per_insert):
        self.samples_per_insert = samples_per_insert
        self.max_avail = self.samples_per_insert * self.tolerance
        cprint(
            f"set sampler per insert: {self.samples_per_insert}, {self.max_avail}",
            "red",
        )

    def want_load(self):
        with self.lock:
            self.size += 1
        return True, "ok"

    def set_avail_max(self):
        with self.lock:
            self.avail = self.max_avail

    @property
    def avail_ratio(self):
        ratio = lambda x, y: x / y if y else np.nan
        return ratio(
            self.avail - self.min_avail,
            self.max_avail - self.min_avail,
        )

    def want_insert(self, x=1):
        with self.lock:
            if self.size >= self.minimum and self.avail >= self.max_avail:
                return (
                    False,
                    f"rate limited: {self.avail:.3f} >= {self.max_avail:.3f}, size: {self.size}",
                )

            self.avail += self.samples_per_insert * x
            self.size += 1
            # cprint(
            # f"[replay] insert at {self.avail} <= {self.max_avail}, size = {self.size}"
            # )
        return True, "ok"

    def want_remove(self):
        with self.lock:
            if self.size < 1:
                return False, "is empty"
            self.size -= 1
        return True, "ok"

    def want_sample(self):
        with self.lock:
            if self.size < self.minimum:
                return False, f"too empty: {self.size} < {self.minimum}"

            if self.avail <= self.min_avail:
                return False, f"rate limited: {self.avail:.3f} <= {self.min_avail:.3f}"

            if not self.overfit_eps:
                self.avail -= 1

        return True, "ok"


class Queue:
    def __init__(self, capacity):
        assert 1 <= capacity
        self.capacity = capacity
        self.size = 0
        self.lock = threading.Lock()

    def want_load(self):
        with self.lock:
            self.size += 1
        return True, "ok"

    def want_insert(self):
        with self.lock:
            if self.size >= self.capacity:
                return False, f"is full: {self.size} >= {self.capacity}"
            self.size += 1
        return True, "ok"

    def want_remove(self):
        with self.lock:
            if self.size < 1:
                return False, "is empty"
            self.size -= 1
        return True, "ok"

    def want_sample(self):
        if self.size < 1:
            return False, "is empty"
        else:
            return True, "ok"
