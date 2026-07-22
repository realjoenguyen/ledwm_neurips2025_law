import json

from messenger.envs import utils


def test_games_json_is_parsed_once_across_splits(monkeypatch, tmp_path):
    source = tmp_path / "games.json"
    source.write_text(
        json.dumps(
            {
                "train_multi_comb": [["airplane", "mage", "dog"]],
                "train_single_comb": [["fish", "bird", "ship"]],
            }
        )
    )
    utils._json_from_path.cache_clear()
    utils._games_from_json.cache_clear()
    original_load = json.load
    calls = []

    def recording_load(handle):
        calls.append(handle.name)
        return original_load(handle)

    monkeypatch.setattr(utils.json, "load", recording_load)
    multi = utils.games_from_json(source, "train_multi_comb")
    single = utils.games_from_json(source, "train_single_comb")
    multi.append("local mutation")

    assert len(calls) == 1
    assert len(single) == 1
    assert len(utils.games_from_json(source, "train_multi_comb")) == 1


def test_text_file_is_read_once(monkeypatch, tmp_path):
    source = tmp_path / "variant.txt"
    source.write_text("variant contents")
    utils._text_from_path.cache_clear()
    original_read_text = utils.Path.read_text
    calls = []

    def recording_read_text(path, *args, **kwargs):
        calls.append(path)
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(utils.Path, "read_text", recording_read_text)

    assert utils.text_from_path(source) == "variant contents"
    assert utils.text_from_path(source) == "variant contents"
    assert calls == [source]
