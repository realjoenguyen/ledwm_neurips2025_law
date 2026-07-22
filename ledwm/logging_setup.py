import logging
import os
import sys

import termcolor
from loguru import logger


LOG_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS Z}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{process.name}:{process.id}</cyan> | "
    "<cyan>{name}:{function}:{line}</cyan> | "
    "{message}"
)

_configured = False
_original_cprint = termcolor.cprint
_WIN_RATE_PREFIX = "actor.win_rate |"


def _env_bool(name, default):
    value = os.environ.get(name)
    if value is None:
        return default
    value = value.strip().lower()
    if value in ("1", "true", "yes", "on"):
        return True
    if value in ("0", "false", "no", "off"):
        return False
    raise ValueError(f"{name} must be a boolean, got {value!r}")


class InterceptHandler(logging.Handler):
    """Route standard-library logging records through Loguru."""

    def emit(self, record):
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        frame = logging.currentframe()
        depth = 0
        while frame and (
            depth == 0 or frame.f_code.co_filename == logging.__file__
        ):
            frame = frame.f_back
            depth += 1
        logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )


def _loguru_cprint(text, color=None, on_color=None, attrs=None, **kwargs):
    """Compatibility adapter for existing termcolor.cprint() call sites."""
    file = kwargs.get("file")
    end = kwargs.get("end", "\n")
    if file not in (None, sys.stdout, sys.stderr) or end not in (None, "\n"):
        return _original_cprint(text, color, on_color, attrs, **kwargs)
    text = str(text)
    if text.startswith(_WIN_RATE_PREFIX):
        logger.opt(depth=1, colors=True).info(
            "<bold><magenta>{}</magenta></bold>", text
        )
    else:
        logger.opt(depth=1).info(text)


def configure_logging():
    """Configure one timestamped application logger per process."""
    global _configured
    if _configured:
        return logger

    level = os.environ.get("LOGURU_LEVEL", "INFO").upper()
    enqueue = _env_bool("LOGURU_ENQUEUE", True)
    colorize = (
        _env_bool("LOGURU_COLORIZE", False)
        if "LOGURU_COLORIZE" in os.environ
        else None
    )

    logger.remove()
    logger.add(
        sys.stderr,
        level=level,
        format=LOG_FORMAT,
        colorize=colorize,
        enqueue=enqueue,
        backtrace=True,
        diagnose=False,
    )
    logging.basicConfig(
        handlers=[InterceptHandler()],
        level={"TRACE": 5, "SUCCESS": 25}.get(
            level, getattr(logging, level, logging.INFO)
        ),
        force=True,
    )
    termcolor.cprint = _loguru_cprint
    _configured = True
    return logger


def complete_logging():
    """Flush queued records before output file descriptors are closed."""
    logger.complete()
