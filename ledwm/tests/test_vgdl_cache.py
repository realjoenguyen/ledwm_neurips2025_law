from pathlib import Path

import vgdl

from vgdl.interfaces.gym.env import VGDLEnv, _parse_domain


def _fixture_paths():
    root = Path(__file__).parents[2]
    stage = root / "messenger-emma" / "messenger" / "envs" / "vgdl_files" / "stage_2"
    game_file = next((stage / "variants").glob("*.txt"))
    level_file = next((stage / "init_states").glob("*.txt"))
    return game_file, level_file


def _make_env(**overrides):
    game_file, level_file = _fixture_paths()
    args = {
        "game_file": game_file,
        "level_file": level_file,
        "notable_sprites": [
            "enemy",
            "message",
            "goal",
            "no_message",
            "with_message",
        ],
        "obs_type": "objects",
        "block_size": 34,
    }
    args.update(overrides)
    return VGDLEnv(**args)


def test_vgdl_domain_is_parsed_once_but_levels_are_fresh(monkeypatch):
    _parse_domain.cache_clear()
    original = vgdl.VGDLParser.parse_game
    calls = []

    def recording_parse(parser, *args, **kwargs):
        calls.append(args[0])
        return original(parser, *args, **kwargs)

    monkeypatch.setattr(vgdl.VGDLParser, "parse_game", recording_parse)
    first = _make_env()
    second = _make_env()

    assert len(calls) == 1
    assert first.game is not second.game
    assert first.game.domain is second.game.domain
    assert first.reset().keys() == second.reset().keys()


def test_vgdl_accepts_inline_game_and_level_descriptions():
    _parse_domain.cache_clear()
    game_file, level_file = _fixture_paths()
    env = _make_env(
        game_file=None,
        level_file=None,
        game_desc=game_file.read_text(),
        level_desc=level_file.read_text(),
    )

    assert env.level_name == "inline"
    assert env.reset()
