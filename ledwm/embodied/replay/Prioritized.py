import threading
import numpy as np
from ledwm.embodied.core.uuid import uuid
from ledwm.embodied.replay import generic, limiters, sampletree, selectors
from collections import defaultdict


class PrioritizedSampler:
    def __init__(
        self,
        config,
        exponent,
        initial,
        branching,
        eps,
        alpha,
        beta,
        c,
        # zero_out_after_sample=False,
        # seed=None,
        min_visit_loss=1,
        model_loss_only=False,
    ):
        self.model_loss_only = model_loss_only
        self.exponent = exponent
        self.initial = initial
        self.tree = sampletree.SampleTree(branching)
        self.key2priority = defaultdict(lambda: self.initial)
        self.key2loss = defaultdict(float)
        self.stepitems = defaultdict(list)
        self.key2visit_count = defaultdict(int)
        # self.zero_out_after_sample = zero_out_after_sample
        # self.c = c
        # self.alpha = alpha
        # self.beta = beta
        self.eps = eps
        self.config = config
        # self.min_visit_loss = 1

    def update_priority(self, key, priority, only_update_in_tree=False):
        if not only_update_in_tree:
            self.key2priority[key] = priority
        self.tree.update(key, priority)

    def has_key(self, key):
        return self.tree.has_key(key)

    def __contains__(self, key):
        return self.has_key(key)

    def update_priorities(
        self,
        model_loss,  # bs
        keys=None,  # bs, 16
        uuid_keys: list = None,
    ):
        if keys is not None:
            assert keys.shape[0] == model_loss.shape[0], (keys.shape, model_loss.shape)
        else:
            assert uuid_keys is not None, uuid_keys
            assert len(uuid_keys) == model_loss.shape[0], (
                len(uuid_keys),
                model_loss.shape,
            )

        if uuid_keys is None:
            keys = generic.convert2uuid(keys)
        else:
            keys = uuid_keys

        for key_id, key in enumerate(keys):
            if self.tree.has_key(key):
                self.update_priority(
                    key,
                    self.calculate_priority(key, model_loss[key_id]),
                )
            else:
                assert self.key2visit_count[key] == 0, self.key2visit_count[key]
                assert self.key2loss[key] == 0, self.key2loss[key]
                assert self.key2priority[key] == self.initial, self.key2priority[key]
                # assert key not in self.key2priority, key

    def calculate_priority(
        self,
        key,
        model_loss,
        # return_loss_only=False,  # after this is sampled then zeroed out the visit loss
    ):
        self.key2loss[key] = model_loss
        if self.model_loss_only:
            return self.get_model_priority(key)
        else:
            return self.get_visit_priority(key) + self.get_model_priority(key)

    def get_visit_priority(self, key):
        return self.initial * np.exp(-self.key2visit_count[key] / (50 * 0.6))

    def get_model_priority(self, key):
        return self.key2loss[key] ** 2

    # def clear_after_batch(self):
    #     self.tree.clear_after_batch()

    def __call__(self):
        key = self.tree.sample()
        self.key2visit_count[key] += 1
        priority = self.calculate_priority(key, self.key2loss[key])
        # avoid resample this key, but key2priority is still the same
        self.update_priority(key, priority, only_update_in_tree=True)
        return key

    def sample_batch(self, size):
        keys = self.tree.sample_batch(size, replace=False)
        assert len(keys) == len(set(keys)), "Prioritized batch contains duplicates."
        for key in keys:
            self.key2visit_count[key] += 1
            priority = self.calculate_priority(key, self.key2loss[key])
            self.update_priority(key, priority, only_update_in_tree=True)
        return keys

    def __setitem__(self, key, steps):
        self.tree.insert(key, self.key2priority[key])

    def __delitem__(self, key):
        if self.tree.has_key(key):
            self.tree.remove(key)
        self.key2priority[key] = self.initial
        self.key2visit_count[key] = 0
        self.key2loss[key] = 0

    def __len__(self):
        return len(self.tree)


class PrioritizedReplay(generic.GenericReplay):
    def __init__(
        self,
        length,
        train_ratio,
        capacity=None,
        directory=None,
        chunks=1024,
        tolerance=1e4,
        min_size=1,
        overfit_eps=False,
        samples_per_insert=None,
        is_eval=False,
        config=None,
        load_directories=None,
        **kwargs,
    ):
        if samples_per_insert:
            limiter = limiters.SamplesPerInsert(
                samples_per_insert, tolerance, min_size, capacity, overfit_eps
            )
        else:
            raise NotImplementedError

        super().__init__(
            batch_length=length,
            overlap=length - 1,
            capacity=capacity,
            remover=selectors.Fifo(),
            sampler=PrioritizedSampler(config=config, **kwargs),
            limiter=limiter,
            directory=directory,
            chunks=chunks,
            train_ratio=train_ratio,
            is_eval=is_eval,
            config=config,
            load_directories=load_directories,
            min_size=min_size,
            # **kwargs,
        )
