"""BRIDGE28: importing an instance must retain its future behavior and identity."""
from copy import deepcopy

import numpy as np
import pytest
import slaythespire as sts
from tests.bridge_fixtures import native_card as _row, native_card_state as _state

pytestmark = [pytest.mark.native, pytest.mark.timeout(30)]


def test_snapshot_preserves_all_native_instance_fields_and_stasis_slots(combat_factory):
    env = combat_factory(['STRIKE_RED'] * 5)
    state = _state(
        hand=[_row('RITUAL_DAGGER', 3, special=39, free=True, retain=True)],
        draw=[_row('SEARING_BLOW', 7, upgrades=3, special=3, base_cost=2, cost=0)],
        discard=[_row('RAMPAGE', 11, special=16, base_cost=0, cost=0)],
        exhaust=[_row('BLOOD_FOR_BLOOD', 20, base_cost=2, cost=0)],
        stasis=[None, _row('BASH', 0, base_cost=2, cost=2)])
    env.thick.bc.set_card_state(state)
    assert env.thick.bc.card_state() == state
    assert env.thick.bc.clone().card_state() == state
    from bridge.schemas.codec_contract import complete_word_tables
    tables = complete_word_tables()
    assert set(state) == set(tables['native_card_snapshot_fields'])
    assert set(state['hand'][0]) == set(tables['native_card_instance_fields'])


def test_restore_preserves_model_inputs_action_masks_and_search_future_key(combat_factory):
    env = combat_factory(['STRIKE_RED'] * 6 + ['DEFEND_RED'] * 4,
                         combat_action_contract='autoregressive-v2', card_language=True)
    battle = env.thick.bc
    state = _state(hand=[_row('BASH', 8, base_cost=0, cost=0),
                         _row('RITUAL_DAGGER', 5, special=39, free=True)],
                   draw=[_row('RAMPAGE', 4, special=21, base_cost=0, cost=0)],
                   discard=[_row('DEFEND_RED', 9, base_cost=0, cost=0)])
    battle.set_card_state(state)
    env._invalidate_backend_cache()
    expected_obs = {key: value.copy() for key, value in env._obs().items()}
    expected_mask = env.action_masks().copy()
    expected_key = battle.nosl_future_key()
    packed = battle.compact_leaf_arrays()
    expected_packed = {key: np.asarray(value).copy() for key, value in packed.items()}
    saved = battle.card_state()
    battle.set_card_state(_state(hand=[_row('STRIKE_RED', 0)]))
    battle.set_card_state(saved)
    env._invalidate_backend_cache()
    assert battle.card_state() == saved
    assert battle.nosl_future_key() == expected_key
    np.testing.assert_array_equal(env.action_masks(), expected_mask)
    for key, value in env._obs().items():
        np.testing.assert_array_equal(value, expected_obs[key], err_msg=key)
    for key, value in battle.compact_leaf_arrays().items():
        np.testing.assert_array_equal(value, expected_packed[key], err_msg=key)


@pytest.mark.parametrize('pile', ['hand', 'draw', 'discard'])
def test_combat_permanent_zero_cost_survives_next_turn_in_every_live_pile(combat_factory, pile):
    env = combat_factory(['STRIKE_RED'] * 5, 'Cultist')
    state = _state(**{pile: [_row('BASH', 4, base_cost=0, cost=0)]})
    env.thick.bc.set_card_state(state)
    env.thick.bc.execute(sts.Action(sts.ActionType.END_TURN))
    after = env.thick.bc.card_state()
    cards = [card for key in ('hand', 'draw', 'discard') for card in after[key]]
    assert len(cards) == 1
    assert cards[0]['uid'] == 4
    assert (cards[0]['base_cost'], cards[0]['cost']) == (0, 0)


def test_free_once_keeps_underlying_cost_and_is_consumed_by_actual_play(combat_factory):
    env = combat_factory(['STRIKE_RED'] * 5, 'Cultist')
    env.thick.bc.set_card_state(_state(hand=[_row('BASH', 4, base_cost=2, cost=2, free=True)]))
    env.thick.bc.apply_custom_state({'player': {'energy': 0}})
    assert env.thick.bc.to_dict()['hand'][0][2] == 0
    action = sts.Action(sts.ActionType.CARD, 0, 0)
    assert env.thick.bc.legal_combat_mask()[0]
    env.thick.bc.execute(action)
    card = env.thick.bc.card_state()['discard'][0]
    assert (card['uid'], card['base_cost'], card['cost'], card['free_to_play_once']) == (4, 2, 2, False)
    assert env.thick.bc.to_dict()['player']['energy'] == 0


@pytest.mark.parametrize('card_id,special,expected_damage', [('RITUAL_DAGGER', 39, 39), ('RAMPAGE', 21, 29)])
def test_current_growth_is_used_in_actual_damage(combat_factory, card_id, special, expected_damage):
    env = combat_factory(['STRIKE_RED'] * 5, 'Cultist')
    env.thick.bc.set_card_state(_state(hand=[_row(card_id, 4, special=special)]))
    env.thick.bc.apply_custom_state({'monsters': [{'hp': 100, 'max_hp': 100, 'block': 0}]})
    env.thick.bc.execute(sts.Action(sts.ActionType.CARD, 0, 0))
    assert env.thick.bc.to_dict()['monsters'][0]['hp'] == 100 - expected_damage


def test_generated_cards_use_restored_cursor_and_do_not_reuse_master_identity(combat_factory):
    env = combat_factory(['STRIKE_RED'] * 5, 'Cultist')
    env.thick.bc.set_card_state(_state(hand=[_row('ANGER', 4, base_cost=0, cost=0)], next_uid=37))
    env.thick.bc.execute(sts.Action(sts.ActionType.CARD, 0, 0))
    state = env.thick.bc.card_state()
    assert {card['uid'] for card in state['discard']} == {4, 37}
    assert state['next_uid'] == 38


@pytest.mark.parametrize('field,value', [
    ('uid', -1), ('uid', 100), ('uid', True), ('upgrades', 2),
    ('special_data', -1), ('special_data', 32768), ('base_cost', -4),
    ('cost', 128), ('cost', '0'), ('free_to_play_once', 1), ('retain', None),
    ('id', 'INVALID'), ('id', 'UNKNOWN'), ('unmapped', 5),
])
def test_invalid_instance_replacement_is_atomic(combat_factory, field, value):
    env = combat_factory(['STRIKE_RED'] * 5)
    before = env.thick.bc.card_state()
    invalid = _state(hand=[_row('BASH', 4)], draw=[_row('STRIKE_RED', 7)])
    invalid['draw'][0][field] = value
    with pytest.raises((ValueError, TypeError, RuntimeError)):
        env.thick.bc.set_card_state(invalid)
    assert env.thick.bc.card_state() == before


@pytest.mark.parametrize('problem', ['missing_field', 'duplicate_uid', 'missing_pile', 'stasis_slots', 'searing_count', 'overflow_hand'])
def test_incomplete_or_inconsistent_snapshot_cannot_replace_cards(combat_factory, problem):
    env = combat_factory(['STRIKE_RED'] * 5)
    before = env.thick.bc.card_state()
    invalid = _state(hand=[_row('BASH', 4)])
    if problem == 'missing_field':
        del invalid['hand'][0]['base_cost']
    elif problem == 'duplicate_uid':
        invalid['discard'] = deepcopy(invalid['hand'])
    elif problem == 'missing_pile':
        del invalid['exhaust']
    elif problem == 'stasis_slots':
        invalid['stasis'] = []
    elif problem == 'searing_count':
        invalid['hand'][0].update(id='SEARING_BLOW', upgrades=3, special_data=0)
    elif problem == 'overflow_hand':
        invalid['hand'] = [_row('BASH', uid) for uid in range(11)]
    with pytest.raises((ValueError, TypeError, RuntimeError)):
        env.thick.bc.set_card_state(invalid)
    assert env.thick.bc.card_state() == before


def test_restored_pain_and_strike_counts_drive_real_effects(combat_factory):
    env = combat_factory(['STRIKE_RED'] * 5, 'Cultist')
    battle = env.thick.bc
    battle.set_card_state(_state(
        hand=[_row('PERFECTED_STRIKE', 10, base_cost=2, cost=2), _row('PAIN', 11, base_cost=-2, cost=-2)],
        draw=[_row('STRIKE_RED', 12)], discard=[_row('POMMEL_STRIKE', 13)],
        exhaust=[_row('STRIKE_RED', 14)]))
    before = battle.to_dict()
    battle.execute(sts.Action(sts.ActionType.CARD, 0, 0))
    after = battle.to_dict()
    assert after['player']['hp'] == before['player']['hp'] - 1
    # Perfected Strike counts itself plus draw/discard Strikes, not exhausted ones.
    assert after['monsters'][0]['hp'] == before['monsters'][0]['hp'] - 12


def test_restored_normality_limits_actual_plays(combat_factory):
    env = combat_factory(['STRIKE_RED'] * 5, 'Cultist')
    battle = env.thick.bc
    battle.set_card_state(_state(hand=[_row('DEFEND_RED', i, base_cost=0, cost=0) for i in range(4)]
                                + [_row('NORMALITY', 5, base_cost=-2, cost=-2)]))
    for _ in range(3):
        assert any(a.action_type == sts.ActionType.CARD for a in battle.legal_actions())
        battle.execute(sts.Action(sts.ActionType.CARD, 0))
    assert not any(a.action_type == sts.ActionType.CARD for a in battle.legal_actions())
    assert len(battle.card_state()['hand']) == 2


@pytest.mark.parametrize('pile', ['hand', 'draw', 'discard'])
def test_restored_blood_card_cache_handles_damage(combat_factory, pile):
    env = combat_factory(['STRIKE_RED'] * 5, 'Cultist')
    battle = env.thick.bc
    state = _state(hand=[_row('BLOODLETTING', 1, base_cost=0, cost=0)],
                   **({pile: [_row('BLOOD_FOR_BLOOD', 4, base_cost=4, cost=4)]} if pile != 'hand' else {}))
    if pile == 'hand':
        state['hand'].append(_row('BLOOD_FOR_BLOOD', 4, base_cost=4, cost=4))
    battle.set_card_state(state)
    battle.execute(sts.Action(sts.ActionType.CARD, 0))
    cards = [c for key in ('hand', 'draw', 'discard') for c in battle.card_state()[key] if c['uid'] == 4]
    assert len(cards) == 1
    assert cards[0]['base_cost'] == cards[0]['cost'] == 3


def test_snapshot_restore_rejects_pending_selection_atomically(combat_factory):
    env = combat_factory(['STRIKE_RED'] * 5, 'Cultist')
    battle = env.thick.bc
    battle.set_card_state(_state(hand=[_row('BURNING_PACT', 1), _row('STRIKE_RED', 2), _row('BASH', 4)],
                                draw=[_row('DEFEND_RED', 3)]))
    battle.execute(sts.Action(sts.ActionType.CARD, 0))
    before, cards = battle.to_dict(), battle.card_state()
    with pytest.raises(ValueError, match='settled player turn'):
        battle.set_card_state(cards)
    assert battle.to_dict() == before
    assert battle.card_state() == cards


def test_restore_preserves_rng_and_native_search_decision(combat_factory):
    env = combat_factory(['STRIKE_RED'] * 5, 'Cultist')
    battle = env.thick.bc
    battle.set_card_state(_state(hand=[_row('BASH', 8, base_cost=0, cost=0)],
                                draw=[_row('DEFEND_RED', 2), _row('STRIKE_RED', 3)],
                                discard=[_row('ANGER', 4, base_cost=0, cost=0)]))
    expected = battle.clone()
    cards = battle.card_state()
    battle.set_card_state(_state(hand=[_row('RAMPAGE', 1)]))
    battle.set_card_state(cards)
    assert battle.nosl_native_search(8, 2, 1.5, 42) == expected.nosl_native_search(8, 2, 1.5, 42)
    for b in (battle, expected):
        b.execute(sts.Action(sts.ActionType.END_TURN))
    assert battle.card_state() == expected.card_state()
    assert battle.to_dict() == expected.to_dict()
