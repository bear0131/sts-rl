"""Caller-owned destination may overlap read-only decoder tables."""
import numpy as np
import pytest

from env.search.native_batch import NativeSearchBatch

pytestmark = [pytest.mark.native, pytest.mark.timeout(60)]


@pytest.mark.parametrize('threads', [1, 4])
@pytest.mark.parametrize('source_layout', ['float32', 'float64', 'negative_stride'])
@pytest.mark.parametrize('stat,field', [('damage', 'hand_damage'), ('block', 'hand_block')])
def test_direct_writer_stats_source_alias_preserves_clear_before_lookup(
        combat_factory, threads, stat, field, source_layout):
    n = 16
    envs = [combat_factory(['STRIKE_RED'] * 10, seed=42+i,
                          combat_action_contract='autoregressive-v2')
            for i in range(n)]
    pool = NativeSearchBatch(envs, threads, direct_observations=True)
    try:
        pool.begin({i: ((np.zeros(e.action_space.n),), dict(
            simulations=4, state_nonce=1729+i, compact_observation=True,
            leaf_batch_size=1)) for i, e in enumerate(envs)})
        target = pool.observations()
        chosen = int(target['hand_ids'][0][target['hand_ids'][0] != 0][0])
        assert pool.rows == n
        shape = pool._card_stats[stat].shape
        prefix = n * 96
        dtype = np.float64 if source_layout == 'float64' else np.float32
        backing = np.full(prefix + int(np.prod(shape)) + n*96, 37, dtype=dtype)
        table = backing[prefix:prefix + int(np.prod(shape))].reshape(shape)
        if source_layout == 'negative_stride':
            table = table[:, ::-1]
        slot = prefix + chosen*shape[1] + (
            shape[1]-1 if source_layout == 'negative_stride' else 0)
        slot *= backing.itemsize // 4
        start = slot - (n-1)*96
        float_words = backing.view(np.float32)
        target['select_card_upgrades'] = float_words[start:start+n*96].reshape(n, 96)
        assert np.shares_memory(table, target['select_card_upgrades'])
        stats = {k: v.copy() for k, v in pool._card_stats.items()}
        stats[stat] = table
        pool.decoder.set_card_stats(stats)
        pool.decoder.decode_native(
            pool.pool, pool._decoder_globals, target, None, pool.language)
        # The last selection row overlaps the stats entry used by every hand.
        # Ordinary combat leaves leave this selection field zero-only. Clearing
        # each row just before decoding would let earlier rows read stale stats.
        # Float64 and reversed views also require checking before forcecasting.
        mask = target['hand_ids'] == chosen
        assert mask.any()
        assert np.all(target[field][mask] == 0), target[field][mask].tolist()
    finally:
        pool.close()


@pytest.mark.parametrize('threads', [1, 4])
@pytest.mark.parametrize('alias', ['global_output', 'zeroed_output'])
def test_direct_writer_global_source_alias_preserves_preparation_order(
        combat_factory, threads, alias):
    n = 16
    envs = [combat_factory(['STRIKE_RED'] * 10, seed=42+i,
                          combat_action_contract='autoregressive-v2')
            for i in range(n)]
    pool = NativeSearchBatch(envs, threads, direct_observations=True)
    try:
        pool.begin({i: ((np.zeros(e.action_space.n),), dict(
            simulations=4, state_nonce=1729+i, compact_observation=True,
            leaf_batch_size=1)) for i, e in enumerate(envs)})
        target = pool.observations()
        globals = dict(pool._decoder_globals)
        expected = np.zeros_like(globals['ascension'])
        if alias == 'global_output':
            shared = np.arange(n+1, dtype=np.float32).reshape(n+1, 1) / 20
            globals['ascension'], target['ascension'] = shared[:-1], shared[1:]
            # Ordered copies propagate the first source row through this view.
        else:
            shared = np.full((n+1, 1), .5, dtype=np.float32)
            target['post_enemy_turn_boundary'] = shared
            globals['ascension'] = shared[1:]
            # Real destination rows are cleared before copying globals. The
            # caller-owned padding row is not cleared and remains a valid source.
            expected[-1] = .5
        pool.decoder.decode_native(
            pool.pool, globals, target, None, pool.language)
        np.testing.assert_array_equal(target['ascension'], expected)
        if alias == 'zeroed_output':
            assert target['post_enemy_turn_boundary'][-1, 0] == .5
    finally:
        pool.close()
