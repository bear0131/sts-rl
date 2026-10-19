"""BRIDGE28: actual real-frame parser -> live search restore -> native actions."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
import slaythespire as sts

from bridge.schemas.card_state import CombatCardIdentityMap, real_card_state
from bridge.schemas.combat_state import normalize_real, real_to_sim_card_enum
from bridge.search_bridge import RealCombatSearch
from tests.bridge_fixtures import real_card, real_combat_state

pytestmark = [pytest.mark.native, pytest.mark.timeout(30)]


def _search(combat_factory, frame):
    env = combat_factory([real_to_sim_card_enum(c['id']) for c in frame['deck']], 'Cultist',
                         combat_action_contract='autoregressive-v2', card_language=True)
    search = RealCombatSearch.__new__(RealCombatSearch)
    search.env = env
    search.adapter = SimpleNamespace(gs=frame, _potions=lambda: ['PAD'] * 3,
                                     select_descriptor=lambda: None)
    search.card_identities = CombatCardIdentityMap()
    search.sim_to_real_targets = {}
    search.run_alignment_checks = 0
    search._reconcile_and_sync(normalize_real(frame))
    return search


def test_live_complete_restore_keeps_cost_identity_and_model_projection(combat_factory):
    master = [real_card('Bash', index=0, base_cost=2), real_card(index=1)]
    bash = real_card('Bash', index=0, base_cost=0, cost=0)
    strike = real_card(index=1, cost=1)
    strike['simulator_card_state'].update(free_to_play_once=True, retain=True)
    search = _search(combat_factory, real_combat_state(deck=master, hand=[strike], draw=[bash]))
    bc = search.env.thick.bc
    state = bc.card_state()
    assert (state['hand'][0]['uid'], state['draw'][0]['uid']) == (1, 0)
    assert state['hand'][0]['cost'] == 1 and state['hand'][0]['free_to_play_once']
    assert state['hand'][0]['retain']
    assert bc.to_dict()['hand'][0][2] == 0
    assert state['draw'][0]['base_cost'] == state['draw'][0]['cost'] == 0
    expected_obs = {k: v.copy() for k, v in search.env._obs().items()}
    expected_mask = search.env.action_masks().copy()
    expected_key = bc.nosl_future_key()
    search._reconcile_and_sync(normalize_real(search.adapter.gs))
    assert bc.card_state() == state and bc.nosl_future_key() == expected_key
    for k, value in search.env._obs().items():
        np.testing.assert_array_equal(value, expected_obs[k], err_msg=k)
    np.testing.assert_array_equal(search.env.action_masks(), expected_mask)
    bc.execute(sts.Action(sts.ActionType.END_TURN))
    bash_after = next(c for c in bc.card_state()['hand'] if c['uid'] == 0)
    assert bash_after['base_cost'] == bash_after['cost'] == 0


@pytest.mark.parametrize('card_id,misc,damage,exhaust', [
    ('RitualDagger', 39, 39, True), ('Rampage', 0, 29, False),
])
def test_live_growth_changes_actual_damage(combat_factory, card_id, misc, damage, exhaust):
    card = real_card(card_id, misc=misc, damage=damage, exhaust=exhaust)
    search = _search(combat_factory, real_combat_state(deck=[card], hand=[card]))
    bc = search.env.thick.bc
    bc.execute(sts.Action(sts.ActionType.CARD, 0, 0))
    assert bc.to_dict()['monsters'][0]['hp'] == 100 - damage


def test_live_free_once_is_not_a_fake_permanent_zero_cost(combat_factory):
    card = real_card('Bash', base_cost=2, cost=2)
    card['simulator_card_state']['free_to_play_once'] = True
    frame = real_combat_state(deck=[real_card('Bash', base_cost=2)], hand=[card])
    frame['combat_state']['player']['energy'] = 0
    search = _search(combat_factory, frame)
    bc = search.env.thick.bc
    assert bc.legal_combat_mask()[0]
    bc.execute(sts.Action(sts.ActionType.CARD, 0, 0))
    row = bc.card_state()['discard'][0]
    assert (row['base_cost'], row['cost'], row['free_to_play_once']) == (2, 2, False)
    assert bc.to_dict()['player']['energy'] == 0


def test_identity_mapping_survives_moves_duplicates_and_master_reorder(combat_factory):
    master = [real_card(index=0), real_card(index=1)]
    copies = [real_card(index=21), real_card(index=20)]
    frame = real_combat_state(deck=master, hand=[master[1], copies[0]], discard=[master[0], copies[1]])
    search = _search(combat_factory, frame)
    state = search.env.thick.bc.card_state()
    assert [c['uid'] for c in state['hand']] == [1, 97]
    assert [c['uid'] for c in state['discard']] == [0, 96]
    frame['deck'].reverse()
    frame['combat_state'].update(hand=[copies[1]], discard_pile=[],
                                 draw_pile=[master[0], copies[0], master[1]])
    search._reconcile_and_sync(normalize_real(frame))
    state = search.env.thick.bc.card_state()
    assert state['hand'][0]['uid'] == 96
    assert [c['uid'] for c in state['draw']] == [1, 97, 0]
    assert state['next_uid'] == 98


def test_live_master_growth_is_refreshed_without_obtain_callbacks(combat_factory):
    card = real_card('RitualDagger', misc=39, damage=39, exhaust=True)
    frame = real_combat_state(deck=[card], exhaust=[card])
    search = _search(combat_factory, frame)
    assert search.env.thick.gc.deck_state()[0]['misc'] == 39
    card['misc'] = 42
    card['simulator_card_state']['base_damage'] = 42
    search._reconcile_and_sync(normalize_real(frame))
    assert search.env.thick.gc.deck_state()[0]['misc'] == 42
    assert search.env.thick.bc.card_state()['exhaust'][0]['special_data'] == 42


def test_live_curse_cost_has_explicit_native_sentinel_translation(combat_factory):
    card = real_card('Pain', base_cost=-2)
    search = _search(combat_factory, real_combat_state(deck=[card], hand=[card]))
    row = search.env.thick.bc.card_state()['hand'][0]
    assert (row['base_cost'], row['cost']) == (-3, -3)
    assert normalize_real(search.adapter.gs)['hand'][0]['cost'] == -2


@pytest.mark.parametrize('field', ['ethereal', 'self_retain'])
def test_unrepresentable_dynamic_card_flags_are_not_silently_dropped(field):
    card = real_card_state(real_card())
    card[field] = True
    mapping = CombatCardIdentityMap()
    with pytest.raises(ValueError, match='dynamic.*not representable'):
        mapping.snapshot([], dict(hand=[card], draw=[], discard=[], exhaust=[]), [None, None])
    assert mapping._temporary == {}


def test_stasis_uses_real_owner_mapping_and_returns_the_same_instance(combat_factory):
    held = real_card('Bash', index=1, base_cost=0, cost=0)
    strike = real_card(index=0)
    frame = real_combat_state(deck=[strike, held], hand=[strike])
    orb = dict(frame['combat_state']['monsters'][0], id='BronzeOrb', current_hp=1,
               powers=[{'id': 'Stasis', 'amount': -1}], move_id=1,
               simulator_stasis_card=held)
    frame['combat_state']['monsters'].append(orb)
    real = normalize_real(frame)
    env = combat_factory(['STRIKE_RED', 'BASH'], 'Automaton')
    search = RealCombatSearch.__new__(RealCombatSearch)
    search.env, search.adapter = env, SimpleNamespace(gs=frame)
    search.card_identities = CombatCardIdentityMap()
    search.sim_to_real_targets = {2: 1}  # Second native orb, not real array index 1.
    snapshot = search._card_state(real)
    assert snapshot['stasis'][0] is None
    assert snapshot['stasis'][1]['uid'] == 1
    env.thick.bc.set_card_state(snapshot)
    env.thick.bc.apply_custom_state({'monsters': [{}, {}, {
        'id': 'BRONZE_ORB', 'hp': 1, 'max_hp': 100, 'powers': [('STASIS', 1)],
        'gone': False, 'move_history': ['BRONZE_ORB_BEAM', 'INVALID']} ]})
    env.thick.bc.execute(sts.Action(sts.ActionType.CARD, 0, 2))
    after = env.thick.bc.card_state()
    assert after['stasis'] == [None, None]
    assert any(c['uid'] == 1 and c['base_cost'] == c['cost'] == 0 for c in after['hand'])


@pytest.mark.parametrize('problem', ['duplicate', 'master_duplicate', 'wrong_master', 'missing_stasis', 'false_stasis'])
def test_real_instance_ambiguity_and_missing_stasis_are_rejected(combat_factory, problem):
    card = real_card()
    frame = real_combat_state(deck=[card], hand=[card])
    if problem == 'duplicate':
        frame['combat_state']['discard_pile'] = [card]
    elif problem == 'master_duplicate':
        frame['deck'].append(card)
    elif problem == 'wrong_master':
        frame['deck'] = [real_card('Bash')]
    elif problem == 'missing_stasis':
        del frame['combat_state']['monsters'][0]['simulator_stasis_card']
    else:
        frame['combat_state']['monsters'][0]['simulator_stasis_card'] = real_card(index=1)
    with pytest.raises(ValueError):
        _search(combat_factory, frame)


def test_strict_native_sync_forbids_lossy_fresh_piles(combat_factory):
    card = real_card()
    search = _search(combat_factory, real_combat_state(deck=[card], hand=[card]))
    spec = search._state_spec(normalize_real(search.adapter.gs))
    bc = search.env.thick.bc
    before = bc.card_state()
    old = deepcopy(spec)
    del old['card_state']
    old.update(hand=[('STRIKE_RED', 0, 1)], draw=[], discard=[], exhaust=[])
    with pytest.raises(ValueError, match='missing card_state'):
        bc.apply_custom_state(old)
    spec['hand'] = []
    with pytest.raises(ValueError, match='cannot be mixed'):
        bc.apply_custom_state(spec)
    assert bc.card_state() == before
