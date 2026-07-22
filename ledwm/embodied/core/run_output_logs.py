import os
import pathlib
import sys
import threading


class FdTee:
    def __init__(self, fd, path):
        self.fd = fd
        self.path = pathlib.Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._original_fd = os.dup(fd)
        self._log_fd = os.open(
            self.path,
            os.O_WRONLY | os.O_CREAT | os.O_APPEND,
            0o644,
        )
        self._read_fd, write_fd = os.pipe()
        os.dup2(write_fd, fd)
        os.close(write_fd)
        self._closed = False
        self._thread = threading.Thread(target=self._pump, daemon=True)
        self._thread.start()

    def _pump(self):
        while True:
            try:
                chunk = os.read(self._read_fd, 8192)
            except OSError:
                break
            if not chunk:
                break
            for target in (self._original_fd, self._log_fd):
                try:
                    os.write(target, chunk)
                except OSError:
                    pass

    def close(self):
        if self._closed:
            return
        self._closed = True
        try:
            if self.fd == 1:
                sys.stdout.flush()
            elif self.fd == 2:
                sys.stderr.flush()
        except Exception:
            pass
        try:
            os.dup2(self._original_fd, self.fd)
        except OSError:
            pass
        self._thread.join(timeout=1.0)
        for fd in (self._read_fd, self._original_fd, self._log_fd):
            try:
                os.close(fd)
            except OSError:
                pass


class RunOutputLogs:
    def __init__(self, logdir):
        self.logdir = pathlib.Path(logdir)
        self.stdout_path = self.logdir / "stdout.log"
        self.stderr_path = self.logdir / "stderr.log"
        self.stdout = FdTee(1, self.stdout_path)
        self.stderr = FdTee(2, self.stderr_path)

    @property
    def paths(self):
        return [self.stdout_path, self.stderr_path]

    def close(self):
        self.stderr.close()
        self.stdout.close()


def install_run_output_logs(logdir):
    return RunOutputLogs(logdir)


def save_run_log_artifact(wandb_module, paths, artifact_name="run-logs"):
    run = getattr(wandb_module, "run", None)
    if run is None:
        return None
    artifact = wandb_module.Artifact(artifact_name, type="logs")
    added = False
    for path in paths:
        path = pathlib.Path(path)
        if not path.exists():
            continue
        artifact.add_file(str(path), name=path.name)
        added = True
    if not added:
        return None
    run.log_artifact(artifact)
    return artifact
