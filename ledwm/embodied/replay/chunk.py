import threading
from datetime import datetime

from termcolor import cprint

import numpy as np

from ledwm.embodied.core.basics import convert
from ledwm.embodied.core.path import Path
from ledwm.embodied.core.uuid import uuid
from ledwm.logging_setup import logger as event_logger


class Chunk:
    def __init__(self, size, successor=None, worker=None):
        now = datetime.now()
        self.time = now.strftime("%Y%m%dT%H%M%S") + f"F{now.microsecond:06d}"
        self.uuid = str(uuid())
        self.successor = successor
        self.worker = worker
        self.size = size
        self.data = None
        self.length = 0
        self.lock = threading.Lock()

    def __repr__(self):
        succ = self.successor or str(uuid(0))
        succ = succ.uuid if isinstance(succ, type(self)) else succ
        return f"Chunk(uuid={self.uuid}, succ={succ}, len={self.length})"

    def __len__(self):
        return self.length

    def __bool__(self):
        return True

    def append(self, step):
        if not self.data:
            example = {k: convert(v) for k, v in step.items()}
            self.data = {
                k: np.empty((self.size,) + v.shape, v.dtype) for k, v in example.items()
            }
        for key, value in step.items():
            self.data[key][self.length] = value
        self.length += 1

    def save(self, directory):
        # The lock makes sure that we aren't trying to save the same chunk multiple
        # times in parallel. This could otherwise happen, for example when a chunk
        # reachings its maximum length soon after initiating a checkpoint write.
        with self.lock:
            succ = self.successor or str(uuid(0))
            succ = succ.uuid if isinstance(succ, type(self)) else succ
            filename = f"{self.time}-{self.uuid}-{succ}-{self.length}.npz"
            filename = Path(directory) / filename
            temporary = Path(directory) / f".{filename.name}.tmp"
            assert self.data is not None, self.data
            # Only save valid data up to self.length, not the full pre-allocated arrays
            data = {k: convert(v[: self.length]) for k, v in self.data.items()}
            data["__replay_worker__"] = np.asarray(
                "" if self.worker is None else str(self.worker)
            )
            with temporary.open("wb") as stream:
                np.savez_compressed(stream, **data)
            temporary.move(filename)
            event_logger.debug(f"replay.chunk_saved | file={filename.name}")
            return filename

    @classmethod
    def load(cls, filename):
        length = int(filename.stem.split("-")[3])
        with Path(filename).open("rb") as f:
            data = np.load(f, allow_pickle=True)
            data = {k: data[k] for k in data.keys()}
        worker = data.pop("__replay_worker__", None)
        chunk = cls(length)
        chunk.time = filename.stem.split("-")[0]
        chunk.uuid = filename.stem.split("-")[1]
        chunk.successor = filename.stem.split("-")[2]
        chunk.length = length
        chunk.data = data
        if worker is not None:
            worker = str(np.asarray(worker).item())
        chunk.worker = worker or None
        return chunk

    @classmethod
    def scan(
        cls,
        directory,
        capacity=None,  # replay.size
        shorten=0,  # batch_length
    ):
        directory = Path(directory)
        filenames, total = [], 0
        filenames_list = []
        for filename in reversed(sorted(directory.glob("*.npz"))):
            parts = filename.stem.split("-")
            if len(parts) != 4:
                continue
            try:
                int(parts[3])
            except ValueError:
                continue
            filenames_list.append(filename)
        # return filenames_list
        for filename in filenames_list:
            if capacity and total >= capacity:
                cprint(f"{total=} epsidoes loaded >= {capacity=}; {len(filenames)=}")
                break
            filenames.append(filename)
            total += max(
                0,
                int(filename.stem.split("-")[3])
                / shorten,  # chunk steps / batch_length = num of data chunks in replay
            )
        # assert len(filenames) > 0, f"Found no files in {directory}"

        print(f"replay.chunk_scan | files={len(filenames)}")
        return sorted(filenames)
