import queue
import threading


_SENTINEL = object()


class AsyncReplayAdder:

    def __init__(self, maxsize=0):
        self._queue = queue.Queue(maxsize=max(0, int(maxsize)))
        self._error = None
        self._closed = False
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def add(self, replay, tran, **kwargs):
        self.raise_if_failed()
        if self._closed:
            raise RuntimeError("AsyncReplayAdder is closed")
        self._queue.put((replay, tran, kwargs))
        self.raise_if_failed()

    def drain(self):
        self._queue.join()
        self.raise_if_failed()

    def close(self):
        if self._closed:
            return
        self._closed = True
        self._queue.put(_SENTINEL)
        self._queue.join()
        self._thread.join(timeout=5)
        self.raise_if_failed()

    def raise_if_failed(self):
        if self._error is not None:
            raise RuntimeError("Async replay add failed") from self._error

    def _run(self):
        while True:
            item = self._queue.get()
            try:
                if item is _SENTINEL:
                    return
                replay, tran, kwargs = item
                replay.add(tran, **kwargs)
            except BaseException as exc:
                if self._error is None:
                    self._error = exc
            finally:
                self._queue.task_done()
