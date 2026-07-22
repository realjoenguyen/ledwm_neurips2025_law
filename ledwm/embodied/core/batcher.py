from functools import partial
import os
import queue as queuelib
import random
import sys
import threading
import traceback
from typing import List

import time
import numpy as np

from ledwm.embodied.replay.generic import GenericReplay


def _env_bool(name, default=False):
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in ("1", "true", "yes", "on")


def _env_float(name, default):
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


class Batcher:
    """Implements zip() with multi-threaded prefetching. The sources are expected to
    yield dicts of Numpy arrays and the iterator will return dicts of batched
    Numpy arrays."""

    def __init__(
        self,
        sources,  # list[generator] = bs; [generator] * bs
        workers=0,
        postprocess=None,
        prefetch_source=4,
        prefetch_batch=2,
        preprocessors=None,
        priority=False,
        timing=False,
        timing_name="batcher",
        timing_every=None,
    ):
        self._workers = workers
        # self.priority = priority
        # if self.priority:
        #     self.lock = threading.Lock()
        #     self.past_sample_ids = []
        self._postprocess = postprocess
        self.preprocessors = preprocessors or {}
        self._timing = timing or _env_bool("LEDWM_BATCHER_TIMING")
        self._timing_name = timing_name
        self._timing_every = (
            _env_float("LEDWM_BATCHER_TIMING_EVERY", 10.0)
            if timing_every is None
            else float(timing_every)
        )
        self._timing_stats = {
            "count": 0,
            "get": 0.0,
            "stack": 0.0,
            "postprocess": 0.0,
            "output_wait": 0.0,
            "total": 0.0,
        }
        self._timing_last = time.time()

        if workers:
            batch_source = self._get_batch_source(sources)
            if batch_source:
                print(
                    f"batcher.mode | source=prioritized_unique_batch | "
                    f"batch_size={len(sources)} | prefetch=1 | "
                    f"handoff=whole_batch",
                    flush=True,
                )
            # Round-robin assign sources to workers.
            self._running = True
            self._threads = []
            self._queues = []
            if batch_source:
                queue = queuelib.Queue(1)
                self._queues.append(queue)
                creator = threading.Thread(
                    target=self._batch_creator,
                    args=(batch_source, queue),
                    daemon=True,
                )
                creator.start()
                self._threads.append(creator)
            else:
                assignments = [([], []) for _ in range(workers)]
                for index, source in enumerate(sources):
                    queue = queuelib.Queue(prefetch_source)
                    self._queues.append(queue)
                    assignment = index % workers
                    assignments[assignment][0].append(source)
                    assignments[assignment][1].append(queue)

                for args in assignments:
                    creator = threading.Thread(
                        target=self._creator, args=args, daemon=True
                    )
                    creator.start()
                    self._threads.append(creator)

            self._batches = queuelib.Queue(1 if batch_source else prefetch_batch)
            batcher = threading.Thread(
                target=self._batcher,
                args=(
                    self._queues,  # all queues from data_loader
                    self._batches,
                    self.preprocessors,
                    bool(batch_source),
                ),
                daemon=True,
            )
            batcher.start()
            self._threads.append(batcher)
        else:
            self._iterators = [source() for source in sources]
            # Create all preprocessors in main thread.
            self.preprocessors = {k: v() for k, v in preprocessors.items()}
        self._once = False

    @staticmethod
    def _get_batch_source(sources):
        if not sources:
            return None
        owners = [getattr(source, "__self__", None) for source in sources]
        owner = owners[0]
        if owner is None or not all(candidate is owner for candidate in owners):
            return None
        if not getattr(owner, "supports_unique_batch_sampling", False):
            return None
        return lambda: owner.sample_batch(len(sources))

    def close(self):
        if self._workers:
            self._running = False
            for thread in self._threads:
                thread.close()

    def __iter__(self):
        if self._once:
            raise RuntimeError(
                "You can only create one iterator per Batcher object to ensure that "
                "data is consumed in order. Create another Batcher object instead."
            )
        self._once = True
        return self

    def __call__(self):
        return self.__iter__()

    def __next__(self):
        if self._workers:  # data_loader
            batch = self._batches.get()
        else:
            # if self.priority:
            #     elems = []
            #     past_sample_ids = []
            #     for x in self._iterators:
            #         elem = next(partial(x, past_sample_ids=past_sample_ids))
            #         elems.append(elem)
            #         past_sample_ids.append(elem["sample_id"])
            # else:

            start = time.time()
            elems = [next(x) for x in self._iterators]  # type: List[GenericReplay.dataset]
            # print("time to get next", time.time() - start)

            start = time.time()
            batch = {}
            for k in elems[0]:
                bx = [x[k] for x in elems]  # get all the values of the key k
                if k in self.preprocessors:
                    preproc = self.preprocessors[k](bx)
                    for preproc_key, preproc_val in preproc.items():
                        batch[f"{k}_{preproc_key}"] = preproc_val
                else:
                    batch[k] = np.stack(bx, 0)
            # print("time to stack", time.time() - start)
            start = 0
            if self._postprocess:
                batch = self._postprocess(batch)
            # print("time to postprocess", time.time() - start)

        if isinstance(batch, Exception):
            raise batch
        return batch

    def _creator(
        self,
        sources,  # list[generator]: generator is GenericReplay.dataset -> sample one data chunk
        outputs,  # queue for each generator
    ):
        # each is data_loader and its queue
        try:
            # iterators = [
            #     (
            #         partial(source, past_sample_ids=self.past_sample_ids)()
            #         if self.priority
            #         else source()
            #     )
            #     for source in sources
            # ]
            batch_source = self._get_batch_source(sources)
            iterators = None if batch_source else [source() for source in sources]

            while self._running:
                if batch_source:
                    items = batch_source()
                else:
                    items = [next(iterator) for iterator in iterators]
                for item, queue in zip(items, outputs):
                    # if self.priority:
                    #     with self.lock:
                    #         self.past_sample_ids.append(item["sample_id"])
                    queue.put(item)

        except Exception as e:
            e.stacktrace = "".join(traceback.format_exception(*sys.exc_info()))
            outputs[0].put(e)
            raise

    def _batch_creator(self, source, output):
        """Prefetch complete unique batches without per-sample queue traffic."""
        try:
            while self._running:
                output.put(source())
        except Exception as e:
            e.stacktrace = "".join(traceback.format_exception(*sys.exc_info()))
            output.put(e)
            raise

    def _record_batcher_timing(self, timings, output):
        if not self._timing:
            return
        stats = self._timing_stats
        stats["count"] += 1
        for key, value in timings.items():
            stats[key] += value
        stats["total"] += sum(timings.values())

        now = time.time()
        if now - self._timing_last < self._timing_every:
            return

        count = max(1, stats["count"])
        parts = [
            f"count={stats['count']}",
            f"get={stats['get'] / count:.4f}s",
            f"stack={stats['stack'] / count:.4f}s",
            f"postprocess={stats['postprocess'] / count:.4f}s",
            f"output_wait={stats['output_wait'] / count:.4f}s",
            f"total={stats['total'] / count:.4f}s",
        ]
        if hasattr(output, "qsize"):
            parts.append(f"queue={output.qsize()}/{output.maxsize}")
        print(
            f"batcher.timing | name={self._timing_name} | " + " | ".join(parts),
            flush=True,
        )

        stats.update(
            {
                "count": 0,
                "get": 0.0,
                "stack": 0.0,
                "postprocess": 0.0,
                "output_wait": 0.0,
                "total": 0.0,
            }
        )
        self._timing_last = now

    def _batcher(
        self,
        sources,  # all queues from data_loader
        output,  #
        preprocessors,  #
        whole_batch_source=False,
    ):
        try:
            # Create preprocessors in-thread.
            preprocessors = {k: v() for k, v in preprocessors.items()}
            while self._running:
                start = time.time()
                if whole_batch_source:
                    elems = sources[0].get()
                    if isinstance(elems, Exception):
                        raise elems
                else:
                    elems = [x.get() for x in sources]
                    random.shuffle(elems)
                get_time = time.time() - start

                for elem in elems:
                    if isinstance(elem, Exception):
                        raise elem

                start = time.time()
                batch = {}
                for k in elems[0]:
                    if k in ["step_time", "id", "game_id"]:
                        continue

                    bx = [x[k] for x in elems]  # get all the values of the key k
                    if k in preprocessors:
                        preproc = preprocessors[k](bx)
                        for preproc_key, preproc_val in preproc.items():
                            batch[f"{k}_{preproc_key}"] = preproc_val
                    else:
                        batch[k] = np.stack(bx, 0)
                stack_time = time.time() - start
                start = time.time()
                if self._postprocess:
                    batch = self._postprocess(batch)
                postprocess_time = time.time() - start
                start = time.time()
                output.put(batch)  # Will wait here if the queue is full.
                output_wait_time = time.time() - start
                self._record_batcher_timing(
                    {
                        "get": get_time,
                        "stack": stack_time,
                        "postprocess": postprocess_time,
                        "output_wait": output_wait_time,
                    },
                    output,
                )

        except Exception as e:
            e.stacktrace = "".join(traceback.format_exception(*sys.exc_info()))
            output.put(e)
            raise
