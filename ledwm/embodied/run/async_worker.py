import queue
import threading


_SENTINEL = object()


class AsyncWorker:

    def __init__(self, name, maxsize=0):
        self._name = name
        self._queue = queue.Queue(maxsize=max(0, int(maxsize)))
        self._error = None
        self._closed = False
        self._thread = threading.Thread(
            target=self._run, name=f"{name}-worker", daemon=True
        )
        self._thread.start()

    def submit(self, fn, *args, **kwargs):
        self.raise_if_failed()
        if self._closed:
            raise RuntimeError(f"{self._name} is closed")
        self._queue.put((fn, args, kwargs))
        self.raise_if_failed()

    def drain(self):
        self._queue.join()
        self.raise_if_failed()

    def close(self, ignore_errors=False):
        if not self._closed:
            self._closed = True
            self._queue.put(_SENTINEL)
            self._queue.join()
            self._thread.join(timeout=5)
        if not ignore_errors:
            self.raise_if_failed()

    def raise_if_failed(self):
        if self._error is not None:
            raise RuntimeError(f"{self._name} failed") from self._error

    def _run(self):
        while True:
            item = self._queue.get()
            try:
                if item is _SENTINEL:
                    return
                fn, args, kwargs = item
                fn(*args, **kwargs)
            except BaseException as exc:
                if self._error is None:
                    self._error = exc
            finally:
                self._queue.task_done()
