import os
import time

from ledwm.embodied.core.run_output_logs import (
    FdTee,
    save_run_log_artifact,
)


def _wait_for_bytes(path, expected, timeout=2.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if path.exists() and path.read_bytes() == expected:
            return
        time.sleep(0.01)
    assert path.read_bytes() == expected


def test_fd_tee_writes_to_original_fd_and_log_file(tmp_path):
    original_read, captured_fd = os.pipe()
    os.set_blocking(original_read, False)
    log_path = tmp_path / "stdout.log"
    tee = FdTee(captured_fd, log_path)

    try:
        os.write(captured_fd, b"hello\n")
        _wait_for_bytes(log_path, b"hello\n")

        deadline = time.time() + 2.0
        data = b""
        while time.time() < deadline:
            try:
                data += os.read(original_read, 1024)
            except BlockingIOError:
                time.sleep(0.01)
                continue
            if data == b"hello\n":
                break

        assert data == b"hello\n"
    finally:
        tee.close()
        os.close(original_read)
        os.close(captured_fd)


def test_save_run_log_artifact_adds_stdout_and_stderr(tmp_path):
    stdout_path = tmp_path / "stdout.log"
    stderr_path = tmp_path / "stderr.log"
    stdout_path.write_text("out")
    stderr_path.write_text("err")

    class Artifact:
        def __init__(self, name, type):
            self.name = name
            self.type = type
            self.files = []

        def add_file(self, path, name=None):
            self.files.append((path, name))

    class Run:
        def __init__(self):
            self.logged = []

        def log_artifact(self, artifact):
            self.logged.append(artifact)

    class Wandb:
        def __init__(self):
            self.run = Run()
            self.Artifact = Artifact

    wandb = Wandb()

    save_run_log_artifact(wandb, [stdout_path, stderr_path])

    artifact = wandb.run.logged[0]
    assert artifact.name == "run-logs"
    assert artifact.type == "logs"
    assert artifact.files == [
        (str(stdout_path), "stdout.log"),
        (str(stderr_path), "stderr.log"),
    ]
