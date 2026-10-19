"""Actual card.exhaust is persistent; queue.exhaustOnUse is a different field."""
import pytest
import slaythespire as sts

from tests.bridge_fixtures import native_card, native_card_state, real_card, real_combat_state
from bridge.schemas.card_state import CombatCardIdentityMap
from bridge.schemas.combat_state import normalize_real

pytestmark = [pytest.mark.native, pytest.mark.timeout(30)]


@pytest.mark.parametrize('card_id,relic', [('PAIN', 'Blue Candle'), ('BURN', 'Medical Kit')])
def test_natural_relic_play_mutates_the_card_and_exhume_keeps_it(combat_factory, card_id, relic):
    env = combat_factory([card_id, 'EXHUME'], 'Cultist', relics=[relic])
    bc = env.thick.bc
    index = next(i for i, c in enumerate(bc.card_state()['hand']) if c['id'] == card_id)
    initial = bc.card_state()['hand'][index]
    assert not initial['exhausts']
    bc.execute(sts.Action(sts.ActionType.CARD, index))
    used = next(c for c in bc.card_state()['exhaust'] if c['id'] == card_id)
    assert used['exhausts'] and used['uid'] == initial['uid']
    index = next(i for i, c in enumerate(bc.card_state()['hand']) if c['id'] == 'EXHUME')
    bc.execute(sts.Action(sts.ActionType.CARD, index))
    recovered = next(c for c in bc.card_state()['hand'] if c['id'] == card_id)
    assert recovered['exhausts'] and recovered['uid'] == initial['uid']


@pytest.mark.parametrize('exhaust', [False, True])
def test_explicit_exhaust_changes_actual_pile_destination(combat_factory, exhaust):
    env = combat_factory(['STRIKE_RED'], 'Cultist')
    bc = env.thick.bc
    state = native_card_state(hand=[native_card('STRIKE_RED', 0, exhaust=exhaust)])
    bc.set_card_state(state)
    bc.execute(sts.Action(sts.ActionType.CARD, 0, 0))
    after = bc.card_state()
    assert len(after['exhaust']) == int(exhaust)
    assert len(after['discard']) == int(not exhaust)
    assert (after['exhaust'] + after['discard'])[0]['exhausts'] == exhaust


def test_dynamic_exhaust_separates_future_keys_and_equivalent_actions(combat_factory):
    env = combat_factory(['STRIKE_RED', 'STRIKE_RED'], 'Cultist',
                         combat_action_contract='autoregressive-v2')
    bc = env.thick.bc
    state = native_card_state(hand=[native_card('STRIKE_RED', 0), native_card('STRIKE_RED', 1)])
    bc.set_card_state(state)
    env._invalidate_backend_cache()
    same_groups = env.equivalent_action_groups().copy()
    key = bc.nosl_future_key()
    state['hand'][1]['exhausts'] = True
    bc.set_card_state(state)
    env._invalidate_backend_cache()
    assert bc.nosl_future_key() != key
    assert (same_groups != env.equivalent_action_groups()).any()
    assert bc.clone().card_state() == state


def test_havoc_one_use_exhaust_does_not_rewrite_the_card_property(combat_factory):
    env = combat_factory(['HAVOC', 'STRIKE_RED'], 'Cultist')
    bc = env.thick.bc
    bc.set_card_state(native_card_state(hand=[native_card('HAVOC', 0)],
                                       draw=[native_card('STRIKE_RED', 1)]))
    bc.execute(sts.Action(sts.ActionType.CARD, 0))
    exhausted = next(c for c in bc.card_state()['exhaust'] if c['id'] == 'STRIKE_RED')
    assert exhausted['exhausts'] is False


@pytest.mark.parametrize('card_id,exhaust,expected', [
    ('BASH', True, True), ('LIMIT_BREAK', True, False), ('DISARM', True, True),
])
def test_upgrade_only_overwrites_exhaust_when_that_upgrade_changes_it(
        combat_factory, card_id, exhaust, expected):
    env = combat_factory(['APOTHEOSIS', card_id], 'Cultist')
    bc = env.thick.bc
    bc.set_card_state(native_card_state(hand=[native_card('APOTHEOSIS', 0, base_cost=2, cost=2)],
                                       draw=[native_card(card_id, 1, exhaust=exhaust)]))
    bc.execute(sts.Action(sts.ActionType.CARD, 0))
    row = bc.card_state()['draw'][0]
    assert row['upgrades'] == 1 and row['exhausts'] == expected


def test_real_used_medical_kit_card_is_not_rejected_or_reset(combat_factory):
    card = real_card('Burn', base_cost=-2, exhaust=True)
    frame = real_combat_state(deck=[], exhaust=[card])
    canonical = normalize_real(frame)
    snapshot = CombatCardIdentityMap().snapshot([], canonical['card_instances'], [None, None])
    env = combat_factory(['STRIKE_RED'], 'Cultist')
    env.thick.bc.set_card_state(snapshot)
    assert env.thick.bc.card_state()['exhaust'][0]['exhausts']


@pytest.mark.parametrize('invalid', [None, 1, 'true'])
def test_exhaust_snapshot_requires_explicit_boolean(combat_factory, invalid):
    env = combat_factory(['STRIKE_RED'], 'Cultist')
    before = env.thick.bc.card_state()
    state = native_card_state(hand=[native_card('STRIKE_RED', 0)])
    state['hand'][0]['exhausts'] = invalid
    with pytest.raises(ValueError, match='explicit booleans'):
        env.thick.bc.set_card_state(state)
    assert env.thick.bc.card_state() == before


def test_old_snapshot_without_actual_exhaust_is_not_filled_with_printed_default(combat_factory):
    env = combat_factory(['STRIKE_RED'], 'Cultist')
    state = native_card_state(hand=[native_card('STRIKE_RED', 0)])
    state['version'] = 1
    del state['hand'][0]['exhausts']
    with pytest.raises(ValueError):
        env.thick.bc.set_card_state(state)
