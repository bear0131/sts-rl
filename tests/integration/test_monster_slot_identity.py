"""F01: observation positions are executable target handles, including gaps."""
import numpy as np
import pytest
import slaythespire as sts

from env.core import action_space as A, encode, run_env, vocab as V
from schemas.combat_state import normalize_sim, _sim_monster_id

pytestmark = [pytest.mark.native, pytest.mark.timeout(30)]

ENCOUNTERS = [('Gremlin Leader', 'GREMLIN_LEADER', 'GREMLIN_LEADER_RALLY'),
              ('Reptomancer', 'REPTOMANCER', 'REPTOMANCER_SUMMON')]
MONSTER_KEYS = ('monster_id', 'monster_move', 'monster_intent_cat',
                'monster_scalars', 'monster_buffs')


@pytest.mark.parametrize('encounter,boss,move', ENCOUNTERS)
@pytest.mark.parametrize('seed', [42, 103])
@pytest.mark.parametrize('stage', ['initial', 'dead_minion', 'summoned'])
def test_native_slots_survive_canonical_packing_and_lifecycle(combat_factory, encounter, boss, move, seed, stage):
    env = combat_factory(['STRIKE_RED'] * 5, encounter, seed=seed)
    bc = env.thick.bc
    original = bc.to_dict()['monsters']
    holes = [i for i, m in enumerate(original) if m['id'] == 'INVALID = 0']
    assert holes
    boss_slot = next(i for i, m in enumerate(original) if m['id'] == boss)
    if stage == 'dead_minion' or (stage == 'summoned' and boss == 'GREMLIN_LEADER'):
        target = next(i for i, m in enumerate(original) if not m['gone'] and m['id'] != boss)
        bc.set_player_strength(999)
        bc.execute(sts.Action(sts.ActionType.CARD, 0, target))
        assert bc.to_dict()['monsters'][target]['gone']
        bc.set_player_strength(0)
    if stage == 'summoned':
        bc.apply_custom_state({'monsters': [{}] * boss_slot + [{'move': move}],
                               'player': {'block': 999}})
        bc.execute(sts.Action(sts.ActionType.END_TURN))
        assert any(bc.to_dict()['monsters'][i]['id'] != 'INVALID = 0' for i in holes)
    env._invalidate_backend_cache()
    raw = bc.to_dict()
    canonical = normalize_sim(raw)
    assert len(canonical['monsters']) == len(raw['monsters'])
    obs = env._obs()
    assert obs['monster_id'][boss_slot] == V.MONSTER_VOCAB_INDEX[_sim_monster_id(boss)]
    language = encode.encode_combat(canonical, combat_language=True)
    packed = env._native_packed_observation_batch({
        k: np.asarray(v)[None, :] for k, v in bc.compact_leaf_arrays().items()})
    assert packed is not None
    for i, monster in enumerate(raw['monsters']):
        if monster['id'] == 'INVALID = 0':
            assert canonical['monsters'][i]['id'] == 'PAD'
            for key in (*MONSTER_KEYS, 'monster_status_language', 'monster_intent_language'):
                assert not np.any(language[key][i]), (key, i)
        else:
            assert obs['monster_id'][i] == V.MONSTER_VOCAB_INDEX[_sim_monster_id(monster['id'])]
            assert obs['monster_scalars'][i, 0] == (not monster['gone'])
        for key in MONSTER_KEYS:
            np.testing.assert_array_equal(obs[key][i], packed[key][0, i], err_msg=f'{key}[{i}]')


@pytest.mark.parametrize('encounter,boss,move', ENCOUNTERS)
@pytest.mark.parametrize('card', ['BASH', 'FLASH_OF_STEEL'])
@pytest.mark.parametrize('path', ['step', 'search'])
def test_observed_target_is_the_monster_that_takes_damage(combat_factory, encounter, boss, move, card, path):
    source = combat_factory([card] * 5, encounter)
    env = source if path == 'step' else source.fork_combat()
    try:
        bc = env.thick.bc
        before = bc.to_dict()
        slot = next(i for i, m in enumerate(before['monsters']) if m['id'] == boss)
        assert env._obs()['monster_id'][slot] == V.MONSTER_VOCAB_INDEX[_sim_monster_id(boss)]
        expected = bc.clone()
        expected.execute(sts.Action(sts.ActionType.CARD, 0, slot))
        if path == 'step':
            env.step(A._PLAY)
            assert env.action_masks()[A._TARGET + slot]
            env.step(A._TARGET + slot)
        else:
            env.search_step(A._PLAY, validate=False)
            env.search_step(A._TARGET + slot, validate=False)
        assert bc.to_dict() == expected.to_dict()
        assert bc.to_dict()['monsters'][slot]['hp'] < before['monsters'][slot]['hp']
        if path == 'search':
            assert source.thick.bc.to_dict() == before
    finally:
        if path == 'search':
            env.close()


def test_old_compacted_native_layout_falls_back_to_canonical(combat_factory, monkeypatch):
    layout = dict(sts.nosl_compact_layout())
    layout.pop('monster_slot_version', None)
    monkeypatch.setattr(run_env, '_NOSL_COMPACT_LAYOUT', layout)
    monkeypatch.setattr(run_env, '_NOSL_COMPACT_LUT', run_env._UNCACHED)
    assert run_env._nosl_compact_lut() is None
    env = combat_factory(['STRIKE_RED'] * 5, 'Gremlin Leader')
    response = env.nosl_is_az_cpp_begin(np.zeros(A.N_ACTIONS), state_nonce=42,
        simulations=2, depth=1, leaf_batch_size=1, compact_observation=True)
    assert not response['done']
    leaves = response['leaves']
    assert leaves
    # This is the corrected Python encoder even when old native packing exists.
    batch = run_env.unpack_nosl_observation_buffer(response)
    assert np.all(batch['monster_id'][:, 3] == V.MONSTER_VOCAB_INDEX['GremlinLeader'])
    env.nosl_is_az_cpp_abort()
