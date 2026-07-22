import concurrent.futures
import time

from termcolor import cprint

from . import basics
from . import path


class Checkpoint:
    def __init__(self, filename=None, parallel=True):
        self._filename = filename and path.Path(filename)
        self._values = {}
        self._parallel = parallel
        if self._parallel:
            self._worker = concurrent.futures.ThreadPoolExecutor(1)
            self._promise = None

    def __setattr__(self, name, value):
        if name in ("exists", "save", "load"):
            return super().__setattr__(name, value)
        if name.startswith("_"):
            return super().__setattr__(name, value)
        has_load = hasattr(value, "load") and callable(value.load)
        has_save = hasattr(value, "save") and callable(value.save)
        if not (has_load and has_save):
            message = f"Checkpoint entry '{name}' must implement save() and load()."
            raise ValueError(message)
        self._values[name] = value

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        try:
            return self._values.get(name)
        except AttributeError:
            raise ValueError(name)

    def exists(self, filename=None):
        assert self._filename or filename
        filename = path.Path(filename or self._filename)
        assert self._filename is not None
        exists = self._filename.exists()
        if exists:
            cprint(f"checkpoint.exists | path={filename} | found=true", "green")
        else:
            cprint(f"checkpoint.exists | path={filename} | found=false", "yellow")
        return exists

    def save(self, filename=None, keys=None):
        assert self._filename or filename
        filename = path.Path(filename or self._filename)
        self.back_up_if_exists(filename)

        if self._parallel:
            if self._promise:
                self._promise.result()
            self._promise = self._worker.submit(self._save, filename, keys)
        else:
            self._save(filename, keys)
        cprint(f"checkpoint.save | path={filename} | keys={keys}")

    def back_up_if_exists(self, filename):
        if filename.exists():
            backup_filename = filename.parent / (filename.name + ".bak")
            if backup_filename.exists():
                backup_filename.remove()
            filename.move(backup_filename)
            cprint(f"checkpoint.backup | path={backup_filename}", "yellow")

    def _save(self, filename, keys):
        keys = tuple(self._values.keys() if keys is None else keys)
        assert all([not k.startswith("_") for k in keys]), keys
        data = {k: self._values[k].save() for k in keys}
        data["_timestamp"] = time.time()
        filename.parent.mkdirs()

        # Write to a temporary file and then atomically rename, so that the
        # requested filename either contains a full checkpoint or does not exist if
        # writing was interrupted.
        tmp = filename.parent / (filename.name + ".tmp")
        tmp.write(basics.pack(data), mode="wb")
        tmp.move(filename)
        print(f"checkpoint.written | path={filename}")

    def load(self, filename=None, keys=None, skip_key="", strict=False, configs=None):
        assert self._filename or filename
        if hasattr(self, "_promise"):
            if self._promise:
                self._promise.result()  # Wait for last save.
        filename = path.Path(filename or self._filename)
        if not filename.exists():
            cprint(f"checkpoint.missing | path={filename}", "red")
            if strict:
                raise FileNotFoundError(filename)
            return

        cprint(f"checkpoint.load | path={filename}", "green")
        data = basics.unpack(filename.read("rb"))
        keys = tuple(data.keys() if keys is None else keys)
        for key in keys:
            if key.startswith("_"):
                continue

            if key not in self._values:
                cprint(
                    f"checkpoint.skip_key | key={key} | reason=missing_in_agent",
                    "yellow",
                )
                continue

            if skip_key != "":
                if skip_key in key:
                    cprint(
                        f"checkpoint.skip_key | key={key} | "
                        f"reason=matched_skip_key | skip_key={skip_key}",
                        "yellow",
                    )
                    continue

            try:
                kw = configs[key] if configs and key in configs else {}
                self._values[key].load(data[key], **kw)
                print(f"checkpoint.key_loaded | key={key}")

            except Exception:
                cprint(f"checkpoint.load_error | key={key}", "red")
                raise

        age = time.time() - data["_timestamp"]
        print(f"checkpoint.loaded | path={filename} | age={age:.0f}s")

    def load_or_save(self):
        if self.exists():
            self.load()
        else:
            self.save()
