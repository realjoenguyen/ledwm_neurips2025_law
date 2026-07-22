import concurrent.futures
import ctypes
import sys
import threading
from typing import List
from termcolor import cprint
import zmq
import time
from collections import deque
import multiprocessing
import cloudpickle
import numpy as np
from . import basics
import concurrent.futures
from ledwm.logging_setup import configure_logging, logger as event_logger


class RemoteError(RuntimeError):
    pass


class ReconnectError(RuntimeError):
    pass


class MaxReconnectionAttemptsError(Exception):
    pass


import socket


def find_free_port(start_port):
    port = int(start_port)
    while True:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("", port))  # Bind to the port
                cprint(f"network.port_check | port={port} | available=true")
                return str(port)
            except socket.error:
                cprint(f"network.port_check | port={port} | available=false", "red")
                port += 1  # If port is in use, increment and try the next one


def find_random_port():
    port = np.random.randint(4000, 6000)
    return find_free_port(port)


MAX_RECONNECT = 15
MAX_RECONNECT_LOAD = 15


def resolve_actor_timeout(args, default=30):
    """Return the actor RPC receive timeout in seconds."""
    timeout = float(getattr(args, "actor_timeout", default))
    if timeout <= 0:
        raise ValueError(f"actor_timeout must be positive, got {timeout}")
    return timeout


class Client:
    def __init__(
        self,
        address,  # actor
        identity=None,
        ipv6=False,
        timeout=10,
        max_reconnect=MAX_RECONNECT,
    ):
        if identity is None:
            identity = np.random.randint(2**32)
        assert isinstance(identity, int), (type(identity), identity)
        self.address = address
        self.identity = identity
        self.ipv6 = ipv6
        self.timeout = timeout
        # self.client_timeout = client_timeout
        self.max_reconnect = max_reconnect
        self.cur_reconnect = 0
        self.socket = None
        self.pending = False
        self.once = True
        self._connect()

    def __call__(self, data):
        assert isinstance(data, dict), type(data)
        if self.pending:
            self._receive()
            # self._receive_with_timeout()
        self.socket.send(basics.pack(data))
        self.once and self._print("Sent first request.")
        self.pending = True
        # return self._receive_with_timeout
        return self._receive

    def _connect(self):
        context = zmq.Context.instance()
        self.socket = context.socket(zmq.REQ)
        self.socket.setsockopt(zmq.IDENTITY, self.identity.to_bytes(16, "big"))
        self.ipv6 and self.socket.setsockopt(zmq.IPV6, 1)
        self.socket.RCVTIMEO = int(1000 * self.timeout)
        address = self._resolve(self.address)
        self._print(f"Client connecting to {address}")
        self.socket.connect(address)
        self.pending = False
        self.once = True

    def _resolve(self, address):
        return f"tcp://{address}"

    def _receive(self):
        try:
            while True:
                received = self.socket.recv()
                self.once and self._print("Received first response.")
                self.once = False
                self.cur_reconnect = 0  # Resetting the reconnect_count

                if received == b"wait":
                    self.socket.send(b"waiting")
                    continue
                else:
                    break

        except zmq.Again:
            self._print(
                f"Reconnecting because server did not respond. Attempt = {self.cur_reconnect}",
                color="red",
            )
            self.socket.close(linger=0)
            self.cur_reconnect += 1

            if self.cur_reconnect >= self.max_reconnect:
                self._print("Max reconnect attempts reached. Exiting.", color="red")
                raise ReconnectError("Max reconnect attempts reached. Timeout")
            self._connect()
            raise ReconnectError()

        result = basics.unpack(received)
        if result.get("type", "data") == "error":
            msg = result.get("message", None)
            raise RemoteError(f"Server responded with an error: {msg}")
        self.pending = False
        return result

    # def _receive_with_timeout(self):
    #     with concurrent.futures.ThreadPoolExecutor() as executor:
    #         future = executor.submit(self._receive)
    #         try:
    #             return future.result(timeout=self.client_timeout)  # Set the timeout
    #         except concurrent.futures.TimeoutError:
    #             self._print(
    #                 "Terminating client as the server did not respond in time.",
    #                 color="red",
    #             )
    #             self.socket.close(linger=0)
    #             # self._connect()  # Reconnect if you wish to keep the client running
    #             raise TimeoutError("Server did not respond in time.")
    #         except Exception as e:
    #             # Handle other types of Exceptions if necessary
    #             raise e

    def _print(self, text, color=None):
        text = f"[{self.identity}] {text}"
        if color:
            event_logger.warning(text)
        else:
            event_logger.debug(text)


class Server:
    def __init__(self, function, port, ipv6=False, batch=-1, threads=1):
        context = zmq.Context.instance()
        self.socket = context.socket(zmq.ROUTER)
        ipv6 and self.socket.setsockopt(zmq.IPV6, 1)
        address = f"tcp://*:{port}"
        event_logger.info("BatchServer listening at {}", address)
        self.socket.bind(address)
        self.function = function
        self.batch = batch
        self.workers = concurrent.futures.ThreadPoolExecutor(threads)
        self.max_inflight = max(1, threads)
        self.promises = deque()
        self.inputs = deque()
        self.requests = {}
        self.outputs = {}
        self.once = True
        self.error = None

    def run(self):
        while True:  # policy loop
            start = time.time()
            self._step()
            duration = time.time() - start
            time.sleep(max(0, 0.001 - duration))

    def _step(self):
        # If there are new messages, dispatch them to the respective queue.
        try:
            while True:
                now = time.time()
                addr, empty, message = self.socket.recv_multipart(zmq.NOBLOCK)
                self.requests[addr] = now
                if message != b"waiting":
                    self.inputs.append((addr, message))
        except zmq.Again:
            pass

        # Reap every completed batch, not only a completed prefix. A slow batch
        # must not keep a later completed batch counted as in flight.
        pending = deque()
        for promise in self.promises:
            if promise.done():
                promise.result()
            else:
                pending.append(promise)
        self.promises = pending

        # Dispatch every complete batch immediately, up to the actual worker
        # concurrency. ThreadPoolExecutor otherwise accepts an unbounded queue,
        # which makes actions stale under a large environment burst.
        batch = max(1, self.batch)
        while len(self.inputs) >= batch and len(self.promises) < self.max_inflight:
            inputs = [self.inputs.popleft() for _ in range(batch)]
            addrs, inputs = [a for a, x in inputs], [x for a, x in inputs]
            self.promises.append(self.workers.submit(self._work, addrs, inputs))
        # If any of the background tasks have set the error field, then wait for
        # all other background tasks to finish first.
        if self.error:
            [x.result() for x in self.promises]

        # Send all available results back to their respective clients. The
        # result is sent instead of the heartbeat, so we remove the heartbeat for
        # the same client from the queue.
        for addr in list(self.outputs.keys()):
            if addr not in self.requests:
                # This can happen if we just sent a heartbeat recently and have not
                # received confirmation from the client yet.
                continue

            message = self.outputs.pop(addr)
            del self.requests[addr]
            # When ROUTER sockets reply to clients that are unreachable, they drop
            # messages by default, which is what we want here.
            # https://zguide.zeromq.org/docs/chapter3/#ROUTER-Error-Handling
            self.socket.send_multipart([addr, b"", message])
            # self.socket.send_multipart([addr, b'', message], zmq.NOBLOCK)

        # Respond with waiting to requests that are older than one second.
        now = time.time()
        for addr, arrival in list(self.requests.items()):
            if now - arrival >= 1.0:
                del self.requests[addr]
                self.socket.send_multipart([addr, b"", b"wait"])
                # self.socket.send_multipart([addr, b'', message], zmq.NOBLOCK)

        # If any of the background tasks have set the error field, raise the
        # error here after we have responded to the clients.
        if self.error:
            raise self.error

    def _work(self, addrs, inputs):
        error = None
        inputs = [basics.unpack(x) for x in inputs]
        # inputs = [inp for inp in inputs if inp.get("type") != "handshake"]
        inputs = {
            k: [inputs[i][k] for i in range(len(inputs))] for k in inputs[0].keys()
        }
        inputs = {
            k: v if isinstance(v[0], str) else np.asarray(v) for k, v in inputs.items()
        }
        if self.batch < 1:
            inputs = {k: v[0] for k, v in inputs.items()}
        try:
            # call callback in parallel_actor
            results = self.function(inputs, [x.hex() for x in addrs])

            if self.batch <= 0:
                results = {k: [v] for k, v in results.items()}
            results = {
                a: {k: v[i] for k, v in results.items()} for i, a in enumerate(addrs)
            }
        except Exception as e:
            error = e
            results = {a: {"type": "error", "message": str(e)} for a in addrs}

        results = {a: basics.pack(v) for a, v in results.items()}
        self.outputs.update(results)

        if error and not self.error:
            self.error = error


class Thread(threading.Thread):
    lock = threading.Lock()

    def __init__(self, fn, *args, name=None):
        self.fn = fn
        self.exitcode = None
        name = name or fn.__name__
        super().__init__(target=self._wrapper, args=args, name=name, daemon=True)

    @property
    def running(self):
        return self.is_alive()

    def _wrapper(self, *args):
        configure_logging()
        try:
            self.fn(*args)
        except Exception:
            with self.lock:
                event_logger.exception("worker.exception | name={}", self.name)
                event_logger.complete()
                self.exitcode = 1
            raise
        self.exitcode = 0

    def terminate(self):
        if not self.is_alive():
            return
        if hasattr(self, "_thread_id"):
            thread_id = self._thread_id
        else:
            thread_id = [k for k, v in threading._active.items() if v is self][0]
        result = ctypes.pythonapi.PyThreadState_SetAsyncExc(
            ctypes.c_long(thread_id), ctypes.py_object(SystemExit)
        )
        if result > 1:
            ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_long(thread_id), None)
        cprint(f"worker.shutdown | type=thread | name={self.name}", "red")
        # cprint("System shutdowns")
        # sys.exit(0)


class Process:
    lock = None
    initializers = []

    def __init__(self, fn, *args, name=None):
        mp = multiprocessing.get_context("spawn")
        if Process.lock is None:
            Process.lock = mp.Lock()
        name = name or fn.__name__
        initializers = cloudpickle.dumps(self.initializers)
        args = (initializers,) + args
        self._process = mp.Process(
            target=self._wrapper, args=(Process.lock, fn, *args), name=name
        )
        self._termination_requested = False
        self._shutdown_reported = False

    def start(self):
        self._process.start()

    @property
    def name(self):
        return self._process.name

    @property
    def running(self):
        return self._process.is_alive()

    @property
    def pid(self):
        return self._process.pid

    @property
    def exitcode(self):
        return self._process.exitcode

    def terminate(self):
        if not self.request_terminate():
            return
        self.join(timeout=1)
        if self.running:
            self.request_kill()
            self.join(timeout=1)
        self.report_shutdown()

    def request_terminate(self):
        proc = self._process
        if not proc.is_alive():
            return False
        self._termination_requested = True
        proc.terminate()
        return True

    def request_kill(self):
        proc = self._process
        if not proc.is_alive():
            return False
        self._termination_requested = True
        proc.kill()
        return True

    def join(self, timeout=None):
        self._process.join(timeout=timeout)

    def report_shutdown(self):
        if self._shutdown_reported or not self._termination_requested:
            return
        self._shutdown_reported = True
        if self.running:
            cprint(
                f"worker.shutdown_timeout | type=process | name={self.name}", "red"
            )
        else:
            cprint(f"worker.shutdown | type=process | name={self.name}", "red")

    def _wrapper(self, lock, fn, *args):
        configure_logging()
        try:
            import cloudpickle

            initializers, *args = args
            for initializer in cloudpickle.loads(initializers):
                initializer()
            fn(*args)
        except KeyboardInterrupt:
            return
        except Exception:
            with lock:
                event_logger.exception("worker.exception | name={}", self.name)
                event_logger.complete()
            raise


def _join_until_stopped(workers, timeout):
    deadline = time.time() + timeout
    pending = [worker for worker in workers if worker.running]
    while pending and time.time() < deadline:
        for worker in pending:
            worker.join(timeout=0)
        pending = [worker for worker in pending if worker.running]
        if pending:
            time.sleep(min(0.05, max(0, deadline - time.time())))
    return [worker for worker in workers if worker.running]


def _terminate_all(workers):
    processes = []
    for worker in workers:
        try:
            if hasattr(worker, "request_terminate"):
                if worker.request_terminate():
                    processes.append(worker)
            else:
                worker.terminate()
        except Exception as e:
            cprint(
                f"worker.terminate_error | name={getattr(worker, 'name', '?')} | "
                f"error={e}",
                "red",
            )

    # All processes receive SIGTERM before we wait on any single process. This
    # keeps Ctrl+C latency bounded when many env workers are blocked in ZMQ.
    stuck = _join_until_stopped(processes, timeout=1)
    for worker in stuck:
        try:
            worker.request_kill()
        except Exception as e:
            cprint(
                f"worker.kill_error | name={getattr(worker, 'name', '?')} | "
                f"error={e}",
                "red",
            )
    _join_until_stopped(stuck, timeout=1)

    for worker in processes:
        worker.report_shutdown()


def run(workers: "List[Thread | Process]"):
    import os
    import signal

    # Turn SIGTERM (e.g. from a job scheduler or `kill <pid>`) into the same
    # clean shutdown path as Ctrl+C so children are never left orphaned.
    def _handle_sigterm(signum, frame):
        raise KeyboardInterrupt

    old_handler = signal.signal(signal.SIGTERM, _handle_sigterm)

    for worker in workers:
        if not worker.running:
            worker.start()

    exit_code = 0
    forced = False
    try:
        while True:
            if all(x.exitcode == 0 for x in workers):
                event_logger.info("worker.shutdown_done | status=success")
                return

            for worker in workers:
                if worker.exitcode not in (None, 0):
                    # Wait for everybody who wants to print their error messages.
                    time.sleep(1)
                    event_logger.error(
                        "worker.crash | name={} | exitcode={}",
                        worker.name,
                        worker.exitcode,
                    )
                    exit_code = 1
                    forced = True
                    return
            time.sleep(0.1)
    except KeyboardInterrupt:
        cprint("\nReceived interrupt, shutting down all workers...", "red")
        forced = True
    finally:
        # Always terminate (and reap) every worker so no ZMQ env process is
        # left behind reconnecting to a dead server.
        _terminate_all(workers)
        signal.signal(signal.SIGTERM, old_handler)
        if forced:
            # After a forced shutdown, non-daemon Server threads (the actor's
            # ThreadPoolExecutor) and the zmq context can block a normal
            # interpreter exit forever -- and that hang ignores Ctrl+C. Every
            # process/worker we own is already terminated, so exit hard instead
            # of hanging on unreachable cleanup.
            cprint("worker.shutdown_done | action=exit", "red")
            event_logger.complete()
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(exit_code)
