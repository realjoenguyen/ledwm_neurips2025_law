import os

import pytest

from ledwm.resume import latest_resumable_run, normalize_resume_flag


def _make_run(root, name, *, checkpoint=True, replay=True, modified=0):
    run = root / name
    (run / "episodes").mkdir(parents=True)
    paths = []
    if checkpoint:
        paths.append(run / "checkpoint_1.ckpt")
    if replay:
        paths.append(run / "episodes" / "replay_bundle.npz")
    for path in paths:
        path.write_bytes(b"state")
        os.utime(path, ns=(modified, modified))
    return run


def test_normalize_resume_flag():
    assert normalize_resume_flag(["--configs", "s1_train", "--resume"]) == [
        "--configs",
        "s1_train",
        "--run.resume",
        "latest",
    ]


def test_latest_resumable_run_requires_checkpoint_and_replay(tmp_path):
    old = _make_run(tmp_path, "old", modified=10)
    expected = _make_run(tmp_path, "complete", modified=20)
    _make_run(tmp_path, "newer-checkpoint-only", replay=False, modified=30)
    _make_run(tmp_path, "newer-replay-only", checkpoint=False, modified=40)

    assert latest_resumable_run(tmp_path) == expected
    assert latest_resumable_run(tmp_path) != old


def test_latest_resumable_run_reports_missing_state(tmp_path):
    _make_run(tmp_path, "checkpoint-only", replay=False, modified=10)

    with pytest.raises(FileNotFoundError, match=r"checkpoint_1\.ckpt.*episodes/\*\.npz"):
        latest_resumable_run(tmp_path)
