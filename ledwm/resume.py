import pathlib


LATEST = "latest"


def normalize_resume_flag(argv):
    """Translate the public bare flag into the existing run.resume setting."""
    normalized = []
    for arg in argv:
        if arg == "--resume":
            normalized.extend(("--run.resume", LATEST))
        else:
            normalized.append(arg)
    return normalized


def latest_resumable_run(run_root, checkpoint_name="checkpoint_1.ckpt"):
    """Return the newest run containing both checkpoint and replay state."""
    run_root = pathlib.Path(run_root)
    if not run_root.is_dir():
        raise FileNotFoundError(f"Resume root does not exist: {run_root}")

    candidates = []
    for run_dir in run_root.iterdir():
        try:
            if not run_dir.is_dir():
                continue
            checkpoint = run_dir / checkpoint_name
            replay = next((run_dir / "episodes").glob("*.npz"), None)
            if checkpoint.is_file() and replay is not None:
                candidates.append(
                    (checkpoint.stat().st_mtime_ns, run_dir.name, run_dir)
                )
        except OSError:
            # A live run can rotate its checkpoint while candidates are scanned.
            continue

    if not candidates:
        raise FileNotFoundError(
            f"No resumable runs in {run_root}; expected {checkpoint_name} "
            "and episodes/*.npz"
        )
    return max(candidates)[2]
