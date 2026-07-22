import concurrent.futures
import hashlib
import threading
from collections import defaultdict, deque
from functools import partial as bind

import numpy as np
from termcolor import cprint
from tqdm import tqdm

from ledwm.embodied.core.path import Path
from ledwm.embodied.core.uuid import uuid

from . import bundle as bundlelib
from . import chunk as chunklib


REPLAY_STEP_ID_KEY = "_replay_step_id"


def replay_step_id(chunk_uuid, index):
    """Stable internal ID derived from persisted chunk provenance."""
    value = f"{chunk_uuid}:{int(index)}".encode("utf-8")
    digest = hashlib.blake2b(value, digest_size=16).digest()
    return np.frombuffer(digest, np.uint8).copy()


class Saver:
    def __init__(
        self, directory, chunks=1024, capacity=None, batch_length=None
    ):
        self.directory = Path(directory)
        self.directory.mkdirs()
        self.chunks = chunks
        self.capacity = capacity
        self.batch_length = batch_length
        self.buffers = defaultdict(bind(chunklib.Chunk, chunks))
        self.workers = concurrent.futures.ThreadPoolExecutor(16)
        self.promises = deque()
        self.loading = False
        self.lock = threading.Lock()
        self.bundle_lock = threading.Lock()

    def _rotate(self, worker):
        buffer = self.buffers[worker]
        if buffer.worker is None:
            buffer.worker = str(worker)
        successor = chunklib.Chunk(self.chunks, worker=buffer.worker)
        buffer.successor = successor
        self.buffers[worker] = successor
        return buffer

    def _collect_finished(self):
        for promise in [x for x in self.promises if x.done()]:
            promise.result()
            self.promises.remove(promise)

    def _select_capacity(self, chunks, capacity, batch_length):
        chunks = sorted(chunks, key=lambda chunk: chunk.time)
        if not capacity:
            return chunks
        shorten = max(1, (batch_length or 1) - 1)
        selected = []
        total = 0
        for chunk in reversed(chunks):
            if total >= capacity:
                break
            selected.append(chunk)
            total += chunk.length / shorten
        return sorted(selected, key=lambda chunk: chunk.time)

    def _load_legacy_chunks(self, filenames):
        if not filenames:
            return []
        threads = min(len(filenames), 32)
        chunks = []
        with concurrent.futures.ThreadPoolExecutor(threads) as executor:
            promises = [
                executor.submit(chunklib.Chunk.load, filename)
                for filename in filenames
            ]
            for filename, promise in zip(filenames, promises):
                try:
                    chunks.append(promise.result())
                except Exception as e:
                    print(f"replay.chunk_load_error | path={filename} | error={e}")
        return chunks

    def _compact(self, dependencies, snapshot_chunks=()):
        for promise in dependencies:
            promise.result()
        with self.bundle_lock:
            filenames = chunklib.Chunk.scan(self.directory, shorten=1)
            if not filenames and not snapshot_chunks:
                return bundlelib.path(self.directory)
            chunks = bundlelib.load(self.directory)
            chunks.extend(self._load_legacy_chunks(filenames))
            chunks.extend(snapshot_chunks)
            chunks = list({chunk.uuid: chunk for chunk in chunks}.values())
            chunks = self._select_capacity(
                chunks, self.capacity, self.batch_length
            )
            filename = bundlelib.save(self.directory, chunks)
            for legacy in filenames:
                try:
                    legacy.remove()
                except FileNotFoundError:
                    pass
            return filename

    def add(self, step, worker):
        if self.loading:
            return None

        with self.lock:
            buffer = self.buffers[worker]
            if buffer.worker is None:
                buffer.worker = str(worker)
            step_id = replay_step_id(buffer.uuid, buffer.length)
            buffer.append(step)
            if buffer.length >= self.chunks:
                buffer = self._rotate(worker)
                self.promises.append(self.workers.submit(buffer.save, self.directory))
            self._collect_finished()
            return step_id

    def save(self, wait=False):
        with self.lock:
            workers = [worker for worker, buffer in self.buffers.items() if buffer.length]
            snapshot_chunks = []
            for worker in workers:
                snapshot_chunks.append(self._rotate(worker))
            dependencies = list(self.promises)
            bundle_promise = self.workers.submit(
                self._compact, dependencies, snapshot_chunks
            )
            self.promises.append(bundle_promise)
        if wait:
            bundle_promise.result()
            with self.lock:
                self._collect_finished()

    def load(self, capacity, batch_length, debug=False):
        filenames = chunklib.Chunk.scan(self.directory, shorten=1)
        chunks = bundlelib.load(self.directory)
        chunks.extend(self._load_legacy_chunks(filenames))
        chunks = list({chunk.uuid: chunk for chunk in chunks}.values())
        chunks = self._select_capacity(chunks, capacity, batch_length)
        if not chunks:
            cprint("replay.load_empty | chunks=0", "red")
            return

        streamids = {}
        for chunk in reversed(sorted(chunks, key=lambda x: x.time)):
            if chunk.worker is not None:
                streamids[chunk.uuid] = chunk.worker
            elif chunk.successor not in streamids:
                streamids[chunk.uuid] = int(uuid())
            else:
                streamids[chunk.uuid] = streamids[chunk.successor]

        self.loading = True
        progress = tqdm(
            total=sum(chunk.length for chunk in chunks),
            desc="replay.load",
            unit="step",
            dynamic_ncols=True,
        )
        try:
            for i, chunk in enumerate(chunks):
                stream = streamids[chunk.uuid]
                for index in range(chunk.length):
                    step = {k: v[index] for k, v in chunk.data.items()}
                    # Keep provenance out of stored model fields while still giving
                    # rebuilt replay sequences the same identity after every resume.
                    step[REPLAY_STEP_ID_KEY] = replay_step_id(chunk.uuid, index)
                    yield step, stream

                progress.update(chunk.length)

                # Free memory early to not require twice the replay capacity.
                chunks[i] = None
                del chunk

                if debug:
                    # stop at i = 10
                    if i >= 10:
                        break
        finally:
            progress.close()
            self.loading = False
