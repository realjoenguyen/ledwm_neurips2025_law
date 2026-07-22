import pytest

from ledwm.embodied.replay.selectors import Fifo


def test_fifo_head_then_arbitrary_removal_keeps_queue_consistent():
    fifo = Fifo()
    table = {}
    for key in "ABCD":
        fifo[key] = None
        table[key] = key

    oldest = fifo()
    del fifo[oldest]
    del table[oldest]

    del fifo["C"]
    del table["C"]

    assert list(fifo.queue) == ["B", "D"]
    assert all(key in table for key in fifo.queue)

    oldest = fifo()
    del fifo[oldest]
    del table[oldest]
    assert fifo() == "D"
    assert fifo() in table


def test_fifo_duplicate_insert_does_not_create_stale_entry():
    fifo = Fifo()

    fifo["A"] = None
    fifo["A"] = None

    assert list(fifo.queue) == ["A"]
    assert len(fifo) == 1
    del fifo["A"]
    with pytest.raises(IndexError, match="FIFO is empty"):
        fifo()
