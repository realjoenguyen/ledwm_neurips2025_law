from pathlib import Path
import importlib
import sys
import types

import numpy as np


def import_messenger_sent(monkeypatch):
    messenger = types.ModuleType("messenger")
    envs = types.ModuleType("messenger.envs")
    two_env_wrapper = types.ModuleType("messenger.envs.TwoEnvWrapper")
    two_env_wrapper.TwoEnvWrapper = object
    encoder_sent = types.ModuleType("ledwm.nets.EncoderSent")
    encoder_sent.NUM_ALL_ENTITIES = 42

    monkeypatch.setitem(sys.modules, "messenger", messenger)
    monkeypatch.setitem(sys.modules, "messenger.envs", envs)
    monkeypatch.setitem(
        sys.modules,
        "messenger.envs.TwoEnvWrapper",
        two_env_wrapper,
    )
    monkeypatch.setitem(sys.modules, "ledwm.nets.EncoderSent", encoder_sent)

    return importlib.import_module("ledwm.embodied.envs.MessengerSent")


def test_mean_sent_embed_path_follows_checkout_root(monkeypatch):
    messenger_sent = import_messenger_sent(monkeypatch)
    fake_file = "/tmp/copied/ledwm/embodied/envs/MessengerSent.py"
    monkeypatch.setattr(messenger_sent, "__file__", fake_file)

    env = object.__new__(messenger_sent.MessengerSent)
    env.t5_sent = False
    env.task = "s1"
    env.mode = "train"
    env.model_sent = "mini"

    assert env.fname() == Path(
        "/tmp/copied/ledwm/embodied/envs/data/messenger/"
        "train_eval_test_s1_mini.pkl"
    )


def test_mean_sent_embed_path_does_not_resolve_symlink_checkout(
    monkeypatch, tmp_path
):
    messenger_sent = import_messenger_sent(monkeypatch)
    real_root = tmp_path / "real"
    real_file = real_root / "ledwm" / "embodied" / "envs" / "MessengerSent.py"
    real_file.parent.mkdir(parents=True)
    real_file.touch()
    link_root = tmp_path / "linked"
    link_root.symlink_to(real_root, target_is_directory=True)

    fake_file = link_root / "ledwm" / "embodied" / "envs" / "MessengerSent.py"
    monkeypatch.setattr(messenger_sent, "__file__", str(fake_file))

    env = object.__new__(messenger_sent.MessengerSent)
    env.t5_sent = False
    env.task = "s1"
    env.mode = "train"
    env.model_sent = "mini"

    assert env.fname() == link_root / (
        "ledwm/embodied/envs/data/messenger/train_eval_test_s1_mini.pkl"
    )


def test_entity_tracking_reaches_s3_episode_length(monkeypatch):
    messenger_sent = import_messenger_sent(monkeypatch)
    env = object.__new__(messenger_sent.MessengerSent)
    env.hist_len = 128
    env.num_entities_task = 3
    env.mask_future_steps_dp = True
    env._step = 0

    def observation(step):
        x = 1 + step % 2
        return {
            "entity_pos": np.array(
                [[x, 1, 1], [3, 2, 2], [4, 3, 3]], dtype=np.int32
            ),
            "avatar_pos": np.array([[0, 0, 1]], dtype=np.int32),
        }

    tracked = env._get_entity_tracking(observation(0), reset=True)
    assert tracked["dp"].shape == (3, env.hist_len)
    assert np.all(tracked["dp"][:, 1:] == -1)

    for step in range(1, 129):
        env._step = step
        tracked = env._get_entity_tracking(observation(step))

    retained_steps = np.arange(1, 129)
    retained_x = 1 + retained_steps % 2
    expected_velocity_x = np.diff(1 + np.arange(129) % 2)

    assert tracked["dp"].shape == (3, env.hist_len)
    np.testing.assert_array_equal(env.entity_pos_hist[:, 0, 0], retained_x)
    np.testing.assert_array_equal(env.entity_vel_hist[:, 0, 0], expected_velocity_x)


def test_entity_tracking_space_uses_configured_history_length(monkeypatch):
    messenger_sent = import_messenger_sent(monkeypatch)
    env = object.__new__(messenger_sent.MessengerSent)
    env.remove_image = True
    env.entity_track = True
    env.num_entities_task = 3
    env.hist_len = 128
    env.num_all_entities = 17
    env.num_sents = 3
    env.use_time_step = False
    env.use_lang = False

    assert env.observation_space["dp"].shape == (3, 128)
