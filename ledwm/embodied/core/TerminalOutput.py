from collections import defaultdict
import numpy as np
import re


class TerminalOutput:
    def __init__(self, pattern=r".*", name=None):
        self._pattern = re.compile(pattern)
        self._name = name
        try:
            import rich.console

            self._console = rich.console.Console()
        except ImportError:
            self._console = None

    # def __call__(self, summaries, filters=["train", "fps", "replay):
    def __call__(self, summaries):
        step = max(s for s, _, _ in summaries)
        scalars = {
            k: float(v)
            for _, k, v in summaries
            if isinstance(v, np.ndarray) and len(v.shape) == 0
        }
        scalars = {k: v for k, v in scalars.items() if self._pattern.search(k)}
        formatted = {k: self._format_value(v) for k, v in scalars.items()}

        if self._console:
            if self._name:
                self._console.rule(f"[green bold]{self._name} (Step {step})")
            else:
                self._console.rule(f"[green bold]Step {step}")

            category = defaultdict(dict)
            for k, v in formatted.items():
                category[k.split("/")[0]].update({k: v})
            for cate, cate_v in category.items():
                self._console.print(
                    " [blue]/[/blue] ".join(f"{k} {v}" for k, v in cate_v.items())
                )
                self._console.print("")

            print("")
        else:
            message = " / ".join(f"{k} {v}" for k, v in formatted.items())
            message = f"[{step}] {message}"
            if self._name:
                message = f"[{self._name}] {message}"
            print(message, flush=True)

    def _format_value(self, value):
        value = float(value)
        if value == 0:
            return "0"

        elif 0.01 < abs(value) < 10000:
            value = f"{value:.2f}"
            value = value.rstrip("0")
            value = value.rstrip("0")
            value = value.rstrip(".")
            return value

        else:
            value = f"{value:.1e}"
            value = value.replace(".0e", "e")
            value = value.replace("+0", "")
            value = value.replace("+", "")
            value = value.replace("-0", "-")
        return value
