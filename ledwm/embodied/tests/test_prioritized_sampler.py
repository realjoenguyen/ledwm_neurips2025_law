from types import SimpleNamespace

from ledwm.embodied.replay.Prioritized import PrioritizedSampler


def make_sampler():
    return PrioritizedSampler(
        config=SimpleNamespace(),
        exponent=1.0,
        initial=100.0,
        branching=16,
        eps=1e-6,
        alpha=1.0,
        beta=1.0,
        c=1.0,
    )


def test_sample_batch_is_unique_and_updates_each_visit_once():
    sampler = make_sampler()
    for key in range(300):
        sampler[key] = None

    keys = sampler.sample_batch(200)

    assert len(keys) == len(set(keys)) == 200
    assert sum(sampler.key2visit_count.values()) == 200
    assert all(sampler.key2visit_count[key] == 1 for key in keys)
    assert all(
        sampler.tree.entries[key].uprob == sampler.get_visit_priority(key)
        for key in keys
    )
