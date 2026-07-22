import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ledwm import jaxutils
from ledwm import ninjax as nj


@pytest.mark.parametrize("unroll", [False, True])
def test_scan_with_output_keeps_output_only_values_out_of_carry(unroll):
    def run(inputs):
        def step(carry, value):
            deter = carry["deter"] + value
            output = {"deter": deter, "diagnostic": deter * 10}
            return {"deter": deter}, output

        return jaxutils.scan_with_output(
            step,
            inputs,
            {"deter": jnp.zeros((1,), jnp.float32)},
            unroll=unroll,
        )

    run = nj.pure(run)
    outputs, _ = run(
        {},
        jax.random.PRNGKey(0),
        jnp.arange(1, 4, dtype=jnp.float32)[:, None],
    )

    np.testing.assert_array_equal(outputs["deter"], [[1], [3], [6]])
    np.testing.assert_array_equal(outputs["diagnostic"], [[10], [30], [60]])


def test_compact_recurrent_state_drops_recorded_output_fields():
    state = {
        "deter": "deter",
        "deter_layers": "layers",
        "stoch": "stoch",
        "logit": "logit",
        "x": "encoded observation",
        "prev_stoch": "previous stochastic state",
        "atten": "attention diagnostics",
    }

    assert jaxutils.compact_rssm_recurrent_state(state) == {
        "stoch": "stoch",
        "deter": "deter",
    }
    assert jaxutils.compact_rssm_recurrent_state(state, gru_multi_layers=True) == {
        "stoch": "stoch",
        "deter_layers": "layers",
    }


def _dense_grid_reference(entity_pos, entity_emb, avatar_pos, avatar_embed, size):
    batch, entities, depth = entity_emb.shape
    grid = jnp.zeros((batch, *size, 17, depth), jnp.float32)
    batch_index, _ = jnp.meshgrid(
        jnp.arange(batch), jnp.arange(entities), indexing="ij"
    )
    entity_index = (
        batch_index.ravel(),
        entity_pos[..., 0].ravel(),
        entity_pos[..., 1].ravel(),
        entity_pos[..., 2].ravel(),
        slice(None),
    )
    grid = grid.at[entity_index].add(entity_emb.reshape((-1, depth)))
    avatar_index = (
        jnp.arange(batch),
        avatar_pos[..., 0].ravel(),
        avatar_pos[..., 1].ravel(),
        avatar_pos[..., 2].ravel(),
        slice(None),
    )
    grid = grid.at[avatar_index].add(avatar_embed[:, 0])
    count = jnp.maximum(jnp.count_nonzero(grid, axis=3), 1)
    return jaxutils.cast_to_compute(jnp.sum(grid, axis=3) / count)


def test_compact_attention_grid_matches_dense_position_slots():
    entity_pos = jnp.array(
        [
            [[1, 1, 2], [1, 1, 3], [2, 2, 4]],
            [[0, 0, 1], [0, 0, 1], [2, 1, 5]],
        ],
        jnp.int32,
    )
    entity_emb = jnp.arange(1, 19, dtype=jnp.float32).reshape((2, 3, 3))
    avatar_pos = jnp.array([[[1, 1, 4]], [[2, 1, 6]]], jnp.int32)
    avatar_embed = jnp.array([[[2.0, 4.0, 6.0]], [[3.0, 6.0, 9.0]]])
    size = (3, 3)

    expected = _dense_grid_reference(
        entity_pos, entity_emb, avatar_pos, avatar_embed, size
    )
    actual = jaxutils.put_atten_to_grid(
        entity_pos, entity_emb, avatar_pos, avatar_embed, size
    )

    assert actual.dtype == jaxutils.COMPUTE_DTYPE
    np.testing.assert_array_equal(actual, expected)


def test_attention_grid_drops_dead_position_outside_10_by_10_grid():
    entity_pos = jnp.array([[[2, 3, 4], [10, 10, 0]]], jnp.int32)
    entity_emb = jnp.array([[[1.0, 2.0], [100.0, 200.0]]])
    avatar_pos = jnp.array([[[10, 10, 0]]], jnp.int32)
    avatar_embed = jnp.array([[[300.0, 400.0]]])

    grid = jaxutils.put_atten_to_grid(
        entity_pos, entity_emb, avatar_pos, avatar_embed, (10, 10)
    )

    assert grid.shape == (1, 10, 10, 2)
    np.testing.assert_array_equal(grid[0, 2, 3], [1.0, 2.0])
    np.testing.assert_array_equal(grid.sum(axis=(1, 2)), [[1.0, 2.0]])


def test_separate_attention_grids_drop_dead_position_outside_10_by_10_grid():
    entity_pos = jnp.array([[[2, 3, 4], [10, 10, 0]]], jnp.int32)
    entity_emb = jnp.array([[[1.0, 2.0], [100.0, 200.0]]])
    avatar_pos = jnp.array([[[1, 1, 5]]], jnp.int32)
    avatar_embed = jnp.array([[[3.0, 4.0]]])

    grids = jaxutils.put_atten_to_grid_seperate(
        entity_pos, entity_emb, avatar_pos, avatar_embed, (10, 10)
    )

    assert grids.shape == (1, 2, 10, 10, 2)
    np.testing.assert_array_equal(grids[0, 0, 2, 3], [1.0, 2.0])
    np.testing.assert_array_equal(grids[0, :, 1, 1], [[3.0, 4.0], [3.0, 4.0]])
    np.testing.assert_array_equal(
        grids.sum(axis=(2, 3)), [[[4.0, 6.0], [3.0, 4.0]]]
    )
