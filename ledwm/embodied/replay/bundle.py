import numpy as np

from ledwm.embodied.core.path import Path
from ledwm.embodied.core.uuid import uuid

from . import chunk as chunklib


FILENAME = "replay_bundle.npz"
FORMAT_VERSION = 2
SUPPORTED_FORMAT_VERSIONS = (1, FORMAT_VERSION)


def path(directory):
    return Path(directory) / FILENAME


def load(directory):
    filename = path(directory)
    if not filename.exists():
        return []
    with filename.open("rb") as stream:
        with np.load(stream, allow_pickle=True) as archive:
            version = int(archive["__format_version__"].item())
            if version not in SUPPORTED_FORMAT_VERSIONS:
                raise ValueError(
                    f"Unsupported replay bundle version {version}: {filename}"
                )
            lengths = archive["__chunk_lengths__"].astype(np.int64)
            times = archive["__chunk_times__"].astype(str)
            uuids = archive["__chunk_uuids__"].astype(str)
            successors = archive["__chunk_successors__"].astype(str)
            workers = (
                archive["__chunk_workers__"].astype(str)
                if version >= 2 and "__chunk_workers__" in archive
                else np.full(len(lengths), "")
            )
            fields = [str(name) for name in archive["__field_names__"]]
            arrays = {
                name: archive[f"field_{index:05d}"]
                for index, name in enumerate(fields)
            }

    total = int(lengths.sum())
    for name, values in arrays.items():
        if len(values) != total:
            raise ValueError(
                f"Replay bundle field {name!r} has {len(values)} steps, expected {total}"
            )

    chunks = []
    offset = 0
    for length, time, chunk_uuid, successor, worker in zip(
        lengths, times, uuids, successors, workers
    ):
        length = int(length)
        chunk = chunklib.Chunk(length)
        chunk.time = str(time)
        chunk.uuid = str(chunk_uuid)
        chunk.successor = str(successor)
        chunk.worker = str(worker) or None
        chunk.length = length
        chunk.data = {
            name: values[offset : offset + length] for name, values in arrays.items()
        }
        chunks.append(chunk)
        offset += length
    return chunks


def save(directory, chunks):
    chunks = sorted(chunks, key=lambda chunk: chunk.time)
    if not chunks:
        return None

    fields = list(chunks[0].data)
    expected = set(fields)
    for chunk in chunks:
        actual = set(chunk.data)
        if actual != expected:
            raise ValueError(
                "Replay chunks have incompatible fields: "
                f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
            )

    successors = []
    for chunk in chunks:
        successor = chunk.successor or str(uuid(0))
        if isinstance(successor, chunklib.Chunk):
            successor = successor.uuid
        successors.append(str(successor))

    payload = {
        "__format_version__": np.asarray(FORMAT_VERSION, np.int64),
        "__chunk_lengths__": np.asarray(
            [chunk.length for chunk in chunks], np.int64
        ),
        "__chunk_times__": np.asarray([chunk.time for chunk in chunks]),
        "__chunk_uuids__": np.asarray([chunk.uuid for chunk in chunks]),
        "__chunk_successors__": np.asarray(successors),
        "__chunk_workers__": np.asarray(
            ["" if chunk.worker is None else str(chunk.worker) for chunk in chunks]
        ),
        "__field_names__": np.asarray(fields),
    }
    for index, name in enumerate(fields):
        payload[f"field_{index:05d}"] = np.concatenate(
            [chunk.data[name][: chunk.length] for chunk in chunks], axis=0
        )

    filename = path(directory)
    temporary = Path(directory) / f".{FILENAME}.{uuid()}.tmp"
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **payload)
    temporary.move(filename)
    steps = sum(chunk.length for chunk in chunks)
    print(
        f"replay.bundle_saved | file={filename.name} | "
        f"chunks={len(chunks)} | steps={steps}"
    )
    return filename
