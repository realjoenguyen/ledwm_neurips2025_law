#!/usr/bin/env python3
from ledwm.configs_util import apply_named_config, load_configs
from ledwm.embodied.core.config import Config


CONFIG_NAMES = (
    "sent",
    "large_encoder_s",
    "large_decoder_s",
    "small_image_data",
    "time",
    "two_cnn",
    "multi_step",
    "no_image_input",
    "sum_reward",
    "prioritize",
    "no_decoder",
    "balanced_weight",
    "medium",
    "reward_onehot",
    "deter_z",
    "s1",
)


def assert_composed_stack_matches_explicit(base, configs, preset, names):
    explicit = base
    for name in names:
        explicit = apply_named_config(explicit, configs, name)
    composed = apply_named_config(base, configs, preset)
    assert composed.flat == explicit.flat


def main():
    configs = load_configs("ledwm")
    base = Config(configs["defaults"])
    assert base["duplicate_pos_eps"] is False
    assert base["run.async_replay_add"] is True
    assert base["run.async_actor_postprocess"] is True
    assert base["run.async_actor_postprocess_queue"] == 2
    assert base["run.actor_timeout"] == 30

    explicit = base
    for name in CONFIG_NAMES:
        explicit = explicit.update(configs[name])
    explicit = explicit.update(
        {
            "batch_length": 25,
            "data_workers": 8,
            "use_wandb": False,
            "num_eval_envs": 30,
            "envs": {"amount": 500},
            "replay": {"size": 10000, "min_size": 64},
            "jax": {
                "allocator": "",
                "prealloc": True,
                "mem_fraction": 0.9,
                "precision": "bfloat16",
            },
            "run": {
                "train_ratio": 512,
                "actor_batch": 64,
                "actor_threads": 8,
                "keep_policy_state_on_device": False,
                "fast_train_metrics": True,
                "fast_optimizer_metrics": True,
                "skip_adam_metrics": True,
                "skip_train_outs": True,
            },
            "test_set": "test",
            "env": {"messenger": {"length": 4}},
            "imag_horizon": 4,
            "rssm": {"deter": 256, "add_h_to_query": False},
            "load_exclude_key": "sent_embed",
        }
    )

    composed = apply_named_config(base, configs, "s1_train")
    assert composed["rssm.image_shape"] == (10, 10)

    keys = [
        "data",
        "encoder_size",
        "decoder_size",
        "two_cnn",
        "use_table",
        "multi_step",
        "has_image_input",
        "replay.type",
        "replay.imbalance",
        "decoder_type",
        "size",
        "reward_head.dist",
        "z_argmax_train",
        "task",
        "rssm.task",
        "batch_length",
        "data_workers",
        "use_wandb",
        "num_eval_envs",
        "envs.amount",
        "replay.size",
        "replay.min_size",
        "jax.allocator",
        "jax.prealloc",
        "jax.mem_fraction",
        "jax.precision",
        "run.train_ratio",
        "run.actor_batch",
        "run.actor_threads",
        "run.keep_policy_state_on_device",
        "run.fast_train_metrics",
        "run.fast_optimizer_metrics",
        "run.skip_adam_metrics",
        "run.skip_train_outs",
        "test_set",
        "env.messenger.length",
        "imag_horizon",
        "rssm.deter",
        "rssm.add_h_to_query",
        "load_exclude_key",
    ]
    for key in keys:
        assert composed[key] == explicit[key], (
            key,
            composed[key],
            explicit[key],
        )

    s2_composed = apply_named_config(base, configs, "s2_train")
    assert s2_composed["env.messenger.length"] == 32
    assert s2_composed["imag_horizon"] == 5
    s2_explicit = base
    for name in ("s2", "sent", "time", "large_encoder_s", "decay_multi_step"):
        s2_explicit = apply_named_config(s2_explicit, configs, name)
    s2_explicit = s2_explicit.update({"decoder_sent": {"cnn_keys": ""}})
    assert s2_composed.flat == s2_explicit.flat
    s3_composed = apply_named_config(base, configs, "s3_train")
    assert s3_composed["task"] == "messenger_s3"
    assert s3_composed["rssm.task"] == "s3"
    assert s3_composed["env.messenger.length"] == 64
    assert s3_composed["envs.amount"] == 10
    assert s3_composed["imag_horizon"] == 5
    assert_composed_stack_matches_explicit(
        base,
        configs,
        "s3_train",
        ("s3", "sent", "time", "large_encoder_s", "decay_multi_step"),
    )

    lwm_base = apply_named_config(base, configs, "lwm")
    assert lwm_base["env.lwm.length"] == 32

    lwm_composed = apply_named_config(base, configs, "lwm_train")
    assert lwm_composed["env.lwm.length"] == 32
    assert lwm_composed["env.lwm.entity_track"] is True
    assert lwm_composed["imag_horizon"] == 32
    assert lwm_composed["rssm.deter"] == 1024
    lwm_explicit = base
    for name in ("lwm", "sent", "time", "large_encoder_s", "decay_multi_step"):
        lwm_explicit = apply_named_config(lwm_explicit, configs, name)
    assert lwm_composed.flat == lwm_explicit.flat
    lwm_small = apply_named_config(base, configs, "lwm_small")
    assert lwm_small["env.lwm.overfit_game"] is True


if __name__ == "__main__":
    main()
