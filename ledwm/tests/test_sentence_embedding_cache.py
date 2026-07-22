import pickle

import numpy as np
import pytest

from ledwm.embodied.envs.sentence_embedding_cache import load_sentence_embeddings


def _write_source(path):
    embeddings = np.arange(24, dtype=np.float32).reshape(6, 4)
    sent2id = {f"sentence-{index}": index for index in range(len(embeddings))}
    with path.open("wb") as handle:
        # Non-T5 Messenger files historically store the array first.
        pickle.dump((embeddings, sent2id), handle)
    return sent2id, embeddings


def test_sentence_embeddings_are_materialized_once_and_memory_mapped(tmp_path):
    source = tmp_path / "embeddings.pkl"
    cache_dir = tmp_path / "cache"
    expected_ids, expected_embeddings = _write_source(source)

    sent2id, embeddings, info = load_sentence_embeddings(
        source, t5_sent=False, cache_dir=cache_dir
    )
    sent2id_again, embeddings_again, info_again = load_sentence_embeddings(
        source, t5_sent=False, cache_dir=cache_dir
    )

    assert sent2id == sent2id_again == expected_ids
    np.testing.assert_array_equal(embeddings, expected_embeddings)
    np.testing.assert_array_equal(embeddings_again, expected_embeddings)
    assert isinstance(embeddings, np.memmap)
    assert isinstance(embeddings_again, np.memmap)
    assert info.created is True
    assert info_again.created is False
    assert info.array_path == info_again.array_path
    with pytest.raises(ValueError):
        embeddings[0, 0] = -1


def test_sentence_embedding_cache_can_be_disabled(tmp_path):
    source = tmp_path / "embeddings.pkl"
    expected_ids, expected_embeddings = _write_source(source)

    sent2id, embeddings, info = load_sentence_embeddings(
        source, t5_sent=False, cache_dir="off"
    )

    assert sent2id == expected_ids
    np.testing.assert_array_equal(embeddings, expected_embeddings)
    assert not isinstance(embeddings, np.memmap)
    assert info.memory_mapped is False
