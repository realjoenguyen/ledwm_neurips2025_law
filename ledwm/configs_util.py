import pathlib

import ruamel.yaml as yaml


INCLUDE_KEY = "_include"
CONFIG_FILES = ("configs.yaml", "s1.yaml", "s2.yaml", "s3.yaml", "lwm.yaml")


def load_configs(config_dir):
    config_dir = pathlib.Path(config_dir)
    loader = yaml.YAML(typ="safe")
    configs = {}

    for filename in CONFIG_FILES:
        path = config_dir / filename
        if not path.exists():
            continue
        data = loader.load(path.read_text()) or {}
        if not isinstance(data, dict):
            raise TypeError(f"{path} must contain a mapping.")
        duplicates = sorted(set(configs) & set(data))
        if duplicates:
            names = ", ".join(duplicates)
            raise KeyError(f"Duplicate configs in {path}: {names}")
        configs.update(data)

    return configs


def _as_include_list(value, name):
    if value is None:
        return ()
    if isinstance(value, str):
        return value.split()
    if isinstance(value, (list, tuple)):
        return tuple(value)
    raise TypeError(f"{name}.{INCLUDE_KEY} must be a string or list.")


def apply_named_config(config, named_configs, name, stack=()):
    if name in stack:
        cycle = " -> ".join((*stack, name))
        raise ValueError(f"Config include cycle: {cycle}")
    if name not in named_configs:
        raise KeyError(f"Unknown config '{name}'.")

    raw = named_configs[name] or {}
    if not isinstance(raw, dict):
        raise TypeError(f"Config '{name}' must be a mapping.")

    includes = _as_include_list(raw.get(INCLUDE_KEY), name)
    for include in includes:
        config = apply_named_config(config, named_configs, include, (*stack, name))

    own_config = {key: value for key, value in raw.items() if key != INCLUDE_KEY}
    if own_config:
        config = config.update(own_config)
    return config
