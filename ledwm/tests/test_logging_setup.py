import logging
import re

from ledwm import logging_setup


def test_cprint_and_stdlib_logging_use_loguru(capsys, monkeypatch):
    monkeypatch.setenv("LOGURU_COLORIZE", "true")
    monkeypatch.setenv("LOGURU_ENQUEUE", "false")
    monkeypatch.setattr(logging_setup, "_configured", False)
    logging_setup.configure_logging()

    logging_setup._loguru_cprint("trainer ready", "green")
    logging.getLogger("dependency").warning("dependency warning")
    logging_setup.complete_logging()

    stderr = capsys.readouterr().err
    plain_stderr = re.sub(r"\x1b\[[0-9;]*m", "", stderr)
    assert " | INFO     | " in plain_stderr
    assert " | trainer ready" in plain_stderr
    assert "\x1b[1mtrainer ready" not in stderr
    assert " | WARNING  | " in plain_stderr
    assert "dependency warning" in plain_stderr


def test_actor_win_rate_is_bold_magenta(capsys, monkeypatch):
    monkeypatch.setenv("LOGURU_COLORIZE", "true")
    monkeypatch.setenv("LOGURU_ENQUEUE", "false")
    monkeypatch.setattr(logging_setup, "_configured", False)
    logging_setup.configure_logging()

    logging_setup._loguru_cprint(
        "actor.win_rate | mode=eval | value=0.7500 | episodes=1000",
        "green",
    )
    logging_setup.complete_logging()

    stderr = capsys.readouterr().err
    plain_stderr = re.sub(r"\x1b\[[0-9;]*m", "", stderr)
    assert "actor.win_rate | mode=eval | value=0.7500 | episodes=1000" in plain_stderr
    assert "\x1b[1m\x1b[35mactor.win_rate" in stderr
