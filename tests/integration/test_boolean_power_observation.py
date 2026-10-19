"""OBS01: actual presence powers reach canonical and packed policy inputs."""
import numpy as np
import pytest
import slaythespire as sts

from bridge.schemas.combat_state import BOOLEAN_PLAYER_POWERS, _player_powers_real
from env.core import encode
from env.core.run_env import _nosl_compact_lut

pytestmark = [pytest.mark.native, pytest.mark.timeout(30)]


def _check(env, name, expected):
    env._invalidate_backend_cache()
    raw = env.thick.bc.to_dict()
    assert dict(raw['player']['powers']).get(name, 0) == expected
    assert env.thick.obs_canonical()['player']['powers'].get(name, 0) == expected
    idx = encode.V.PLAYER_BUFF_VOCAB_INDEX[name]
    want = np.float32(expected / encode.BUFF_SCALE)
    assert env._obs()['player_buffs'][idx] == want
    packed = env.thick.bc.compact_leaf_arrays()
    status = list(sts.nosl_compact_layout()['player_status_names']).index(name)
    assert packed['scalars'][6 + status] == expected
    if env._card_language:
        return  # The real search-leaf check below includes its language arrays.
    batch = env._native_packed_observation_batch({
        'categorical': np.asarray(packed['categorical'])[None],
        'scalars': np.asarray(packed['scalars'])[None],
    })
    assert batch['player_buffs'][0, idx] == want


@pytest.mark.parametrize('name', sorted(BOOLEAN_PLAYER_POWERS))
def test_presence_not_numeric_map_and_clear(combat_factory, name):
    env = combat_factory(['STRIKE_RED'] * 5)
    env.thick.bc.set_player_status(name.upper().replace(' ', '_'), 7)
    _check(env, name, 1)
    env.thick.bc.set_player_status(name.upper().replace(' ', '_'), 0)
    _check(env, name, 0)


@pytest.mark.parametrize('card,name', [('CORRUPTION', 'Corruption'), ('BARRICADE', 'Barricade')])
def test_actual_power_play_reaches_search_leaves(combat_factory, card, name):
    env = combat_factory([card, 'DEFEND_RED', 'STRIKE_RED'], card_language=True)
    battle = env.thick.bc
    battle.apply_custom_state({'hand': [card, 'DEFEND_RED', 'STRIKE_RED'],
                              'player': {'energy': 9}})
    _check(env, name, 0)
    battle.execute(sts.Action(sts.ActionType.CARD, 0))
    _check(env, name, 1)
    logits = np.zeros(181)
    logits[1] = 50  # Defend, without advancing to enemy phase.
    args = dict(simulations=1, depth=1, nonce=42)
    normal = battle.nosl_neural_search_begin(logits, **args).next_batch(1)[0]
    packed = battle.nosl_neural_search_begin(logits, **args).next_batch_packed(
        1, _nosl_compact_lut()['language_card_index'])
    expected = env._native_leaf_observation(normal['state'], combat_only=True)
    actual = env._native_packed_observation_batch(packed)
    idx = encode.V.PLAYER_BUFF_VOCAB_INDEX[name]
    assert expected['player_buffs'][idx] == np.float32(1 / encode.BUFF_SCALE)
    for key in expected:
        np.testing.assert_array_equal(expected[key], actual[key][0], err_msg=key)


def test_snecko_eye_exports_real_confused_bit(combat_factory):
    env = combat_factory(['STRIKE_RED'] * 5, relics=['Snecko Eye'])
    _check(env, 'Confused', 1)


def test_numeric_powers_keep_magnitude_and_negative_strength(combat_factory):
    env = combat_factory(['STRIKE_RED'] * 5)
    battle = env.thick.bc
    battle.apply_custom_state({'hand': ['FLEX']})
    battle.execute(sts.Action(sts.ActionType.CARD, 0))
    battle.set_player_strength(0)
    battle.execute(sts.Action(sts.ActionType.END_TURN))
    env.thick.bc.set_player_status('VIGOR', 7)
    _check(env, 'Strength', -2)
    _check(env, 'Vigor', 7)
    env.thick.bc.set_player_status('VIGOR', 0)
    _check(env, 'Vigor', 0)


def test_real_bridge_presence_and_native_contract():
    powers = _player_powers_real([{'id': name, 'amount': -1}
                                 for name in BOOLEAN_PLAYER_POWERS]
                                + [{'id': 'Strength', 'amount': -3}])
    assert all(powers[name] == 1 for name in BOOLEAN_PLAYER_POWERS)
    assert powers['Strength'] == -3
    assert encode.OBS_VERSION == sts.nosl_compact_layout()['observation_version'] == 20
    assert sts.nosl_compact_layout()['version'] == 6
