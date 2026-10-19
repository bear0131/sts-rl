"""BRIDGE27: live branch globals must override persistent root metadata."""
import numpy as np
import pytest
import slaythespire as sts

from env.core import encode
from env.search.native_batch import NativeSearchBatch

pytestmark = [pytest.mark.native, pytest.mark.timeout(30)]


def _decode(env, decoder, *, packed_gold=None):
    raw = env.thick.bc.compact_leaf_arrays()
    packed = {
        'categorical': np.asarray(raw['categorical'])[None],
        'scalars': np.asarray(raw['scalars'])[None],
        'owners': np.array([0], dtype=np.int64),
        'selection_prefix': np.array([0], dtype=np.int32),
        'legal_policy_ids': np.empty((1, 0), dtype=np.int32),
    }
    if packed_gold is not None:
        packed['scalars'][0, 5] = packed_gold
    if decoder == 'python':
        return env._native_packed_observation_batch(packed)
    pool = NativeSearchBatch([env], 1)
    try:
        globals = {key: value[None] for key, value in env._native_packed_global_fields().items()}
        return pool.decoder.decode(packed, globals)
    finally:
        pool.close()


@pytest.mark.parametrize('decoder', ['python', 'native'])
@pytest.mark.parametrize('gold', [0, 147])
def test_leaf_gold_uses_current_branch_not_root_journal(combat_factory, decoder, gold):
    env = combat_factory(['STRIKE_RED'] * 5)
    env.thick.gc.gold = 99
    env.thick.bc.apply_custom_state({'player': {'gold': gold}})
    canonical = env._native_leaf_observation(env.thick.bc.to_dict(), combat_only=True)
    assert canonical['gold'][0] == np.float32(gold / encode.GOLD_SCALE)
    assert _decode(env, decoder)['gold'][0, 0] == canonical['gold'][0]
    assert env.thick.gc.gold == 99, 'hypothetical branches must not mutate the root journal'


@pytest.mark.parametrize('decoder', ['python', 'native'])
@pytest.mark.parametrize('upgrade,reward', [(0, 20), (1, 25)])
def test_hand_of_greed_branch_observation_includes_actual_gold_reward(combat_factory, decoder, upgrade, reward):
    env = combat_factory(['HAND_OF_GREED'] * 5)
    env.thick.gc.gold = 99
    env.thick.bc.apply_custom_state({
        'player': {'gold': 99, 'energy': 3},
        'hand': [('HAND_OF_GREED', upgrade)], 'monsters': [{'hp': 1}],
    })
    env.thick.bc.execute(sts.Action(sts.ActionType.CARD, 0, 0))
    assert env.thick.bc.to_dict()['player']['gold'] == 99 + reward
    assert _decode(env, decoder)['gold'][0, 0] == np.float32((99 + reward) / encode.GOLD_SCALE)


@pytest.mark.parametrize('decoder', ['python', 'native'])
@pytest.mark.parametrize('gold', [-1, 1.5, float('nan'), 32768])
def test_invalid_packed_gold_is_rejected_instead_of_copying_root(combat_factory, decoder, gold):
    env = combat_factory(['STRIKE_RED'] * 5)
    with pytest.raises((ValueError, RuntimeError), match='gold'):
        _decode(env, decoder, packed_gold=gold)


def test_compact_player_scalar_word_table_matches_native_metadata():
    from bridge.schemas.codec_contract import complete_word_tables
    table = complete_word_tables()['compact_player_scalar_fields']
    assert tuple(table) == tuple(sts.nosl_compact_layout()['player_scalar_fields'])
    assert [row['index'] for row in table.values()] == list(range(6))
    assert table['gold']['index'] == 5
