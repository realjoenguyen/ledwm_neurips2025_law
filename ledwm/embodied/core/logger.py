import concurrent.futures
import time
from typing import List
import numpy as np
from typing import TYPE_CHECKING
from termcolor import cprint
from ledwm.embodied.core.counter import Counter

if TYPE_CHECKING:
    from ledwm.embodied.core.TerminalOutput import TerminalOutput
    from ledwm.embodied.core.TensorBoardOutput import TensorBoardOutput
    from ledwm.embodied.core.WandBOutput import WandBOutput


class AsyncOutput:
    def __init__(self, callback, parallel=True):
        self._callback = callback
        self._parallel = parallel
        if parallel:
            self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
            self._future = None

    def __call__(self, summaries):
        if self._parallel:
            self._future and self._future.result()
            self._future = self._executor.submit(self._callback, summaries)
        else:
            self._callback(summaries)


class Logger:
    def __init__(
        self,
        step: "Counter",
        outputs: "List[AsyncOutput | TerminalOutput | WandBOutput, TensorBoardOutput]",
        multiplier=1,
    ):
        assert outputs, "Provide a list of logger outputs."
        self.step = step
        self.outputs = outputs
        self.multiplier = multiplier
        self._last_step = None
        self._last_time = None
        self._metrics = []

    def update_step(self, step):
        self.step.load(step)

    def add(self, mapping: dict, prefix=None):
        step = int(self.step) * self.multiplier

        for name, value in dict(mapping).items():
            name = f"{prefix}/{name}" if prefix else name
            has_nan = False
            is_inf = False
            if isinstance(value, str):
                pass

            else:
                # turn to np in logger
                value = np.asarray(value)
                if len(value.shape) not in (0, 1, 2, 3, 4):
                    raise ValueError(
                        f"Shape {value.shape} for name '{name}' cannot be "
                        "interpreted as scalar, vector, image, or video."
                    )
                has_nan = np.isnan(value).any()
                is_inf = np.isinf(value).any()

            if not has_nan and not is_inf:
                self._metrics.append((step, name, value))

            # else:
            #     cprint(f"Skipping NaN or Inf value for {name}", "red")

    def scalar(self, name, value):
        value = np.asarray(value)
        assert len(value.shape) == 0, value.shape
        self.add({name: value})

    def vector(self, name, value):
        value = np.asarray(value)
        assert len(value.shape) == 1, value.shape
        self.add({name: value})

    def image(self, name, value):
        value = np.asarray(value)
        assert len(value.shape) in (2, 3), value.shape
        self.add({name: value})

    def video(self, name, value):
        value = np.asarray(value)
        assert len(value.shape) == 4, value.shape
        self.add({name: value})

    def text(self, name, value):
        assert isinstance(value, str), (type(value), str(value)[:100])
        self.add({name: value})

    def write(self, fps=False, real_step=True):
        if fps:
            value = self._compute_fps()
            if value is not None:
                self.scalar("fps", value)
        if not self._metrics:
            return
        for output in self.outputs:
            output.__call__(tuple(self._metrics))
        self._metrics.clear()

    def _compute_fps(self):
        step = int(self.step) * self.multiplier

        if self._last_step is None:
            self._last_time = time.time()
            self._last_step = step
            return None
        steps = step - self._last_step

        duration = time.time() - self._last_time
        self._last_time += duration
        self._last_step = step
        return steps / duration

    def __repr__(self):
        return f"Logger(step={self.step}, outputs={self.outputs}, multiplier={self.multiplier})"
