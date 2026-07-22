from ledwm.embodied.core import path
from ledwm.embodied.core.logger import AsyncOutput


import numpy as np


import collections
import json
import re


class JSONLOutput(AsyncOutput):
    def __init__(
        self,
        logdir,
        filename="metrics.jsonl",
        pattern=r".*",
        strings=False,
        parallel=True,
    ):
        super().__init__(self._write, parallel)
        self._filename = filename
        self._pattern = re.compile(pattern)
        self._strings = strings
        self._logdir = path.Path(logdir)
        self._logdir.mkdirs()

    def _write(self, summaries):
        bystep = collections.defaultdict(dict)
        for step, name, value in summaries:
            if not self._pattern.search(name):
                continue
            if isinstance(value, str) and self._strings:
                bystep[step][name] = value
            if isinstance(value, np.ndarray) and len(value.shape) == 0:
                bystep[step][name] = float(value)
        lines = "".join(
            [
                json.dumps({"step": step, **scalars}) + "\n"
                for step, scalars in bystep.items()
            ]
        )
        with (self._logdir / self._filename).open("a") as f:
            f.write(lines)
