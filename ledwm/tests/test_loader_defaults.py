import ast
import pathlib

from ledwm.configs_util import apply_named_config, load_configs
from ledwm.embodied.core.config import Config


REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]


def _jaxagent_dataset_default(name):
    tree = ast.parse((REPO_ROOT / "ledwm" / "jaxagent.py").read_text())
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "JAXAgent":
            for child in node.body:
                if isinstance(child, ast.FunctionDef) and child.name == "dataset":
                    args = child.args.args[-len(child.args.defaults) :]
                    defaults = dict(zip((arg.arg for arg in args), child.args.defaults))
                    return ast.literal_eval(defaults[name])
    raise AssertionError("JAXAgent.dataset default not found")


def test_s1_train_uses_sixteen_data_workers_by_default():
    named_configs = load_configs(REPO_ROOT / "ledwm")
    config = Config(named_configs["defaults"])
    config = apply_named_config(config, named_configs, "s1_train")

    assert config.data_workers == 16


def test_jaxagent_dataset_prefetches_four_batches_by_default():
    assert _jaxagent_dataset_default("prefetch_batch") == 4


def test_episode_logging_is_disabled_by_default():
    named_configs = load_configs(REPO_ROOT / "ledwm")
    config = Config(named_configs["defaults"])

    assert config.run.episode_log_every == 0


def test_jax_speed_defaults_use_bfloat16_and_autotune_level_four():
    named_configs = load_configs(REPO_ROOT / "ledwm")
    config = Config(named_configs["defaults"])

    assert config.jax.precision == "bfloat16"
    assert config.jax.xla_autotune_level == 4
