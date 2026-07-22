import subprocess
import sys
from pathlib import Path


def test_train_script_bootstraps_local_packages_without_pythonpath(tmp_path):
    script = Path(__file__).parents[1] / "train.py"
    code = (
        "import runpy; "
        f"runpy.run_path({str(script)!r}, run_name='ledwm_train_import_probe'); "
        "import messenger.envs.config"
    )

    result = subprocess.run(
        [sys.executable, "-I", "-c", code],
        cwd=tmp_path,
    )

    assert result.returncode == 0


def test_train_module_import_does_not_import_jax():
    code = (
        "import sys; "
        "import ledwm.train; "
        "raise SystemExit(1 if 'jax' in sys.modules else 0)"
    )

    result = subprocess.run([sys.executable, "-c", code])

    assert result.returncode == 0


def test_parallel_env_worker_import_does_not_import_jax():
    code = (
        "import sys; "
        "import ledwm.embodied.run.env_worker; "
        "raise SystemExit(1 if 'jax' in sys.modules else 0)"
    )

    result = subprocess.run([sys.executable, "-c", code])

    assert result.returncode == 0


def test_messenger_sent_import_does_not_import_jax():
    code = (
        "import sys; "
        "import ledwm.embodied.envs.MessengerSent; "
        "raise SystemExit(1 if 'jax' in sys.modules else 0)"
    )

    result = subprocess.run([sys.executable, "-c", code])

    assert result.returncode == 0
