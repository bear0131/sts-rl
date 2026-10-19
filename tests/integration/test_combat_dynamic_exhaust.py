"""Actual card.exhaust is persistent; queue.exhaustOnUse is a different field."""
import pytest
import slaythespire as sts

from tests.bridge_fixtures import native_card, native_card_state

pytestmark = [pytest.mark.native, pytest.mark.timeout(30)]


@pytest.mark.parametrize('card_id,relic', [('PAIN', 'Blue Candle'), ('BURN', 'Medical Kit')])
def test_natural_relic_play_mutates_the_card_and_exhume_keeps_it(combat_factory, card_id, relic):
    env = combat_factory(['STRIKE_RED'], 'Cultist', relics=[relic])
    bc = env.thick.bc
    # Status cards belong in combat piles, not the persistent master deck.
    bc.apply_custom_state({'hand': [(card_id, 0), ('EXHUME', 0)],
                           'draw': [], 'discard': [], 'exhaust': []})
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
    ('THINKING_AHEAD', True, False),
])
def test_upgrade_only_overwrites_exhaust_when_that_upgrade_changes_it(
        combat_factory, card_id, exhaust, expected):
    env = combat_factory(['APOTHEOSIS', card_id], 'Cultist')
    bc = env.thick.bc
    bc.set_card_state(native_card_state(hand=[native_card('APOTHEOSIS', 0, base_cost=2, cost=2, exhaust=True)],
                                       draw=[native_card(card_id, 1, exhaust=exhaust)]))
    bc.execute(sts.Action(sts.ActionType.CARD, 0))
    row = bc.card_state()['draw'][0]
    assert row['upgrades'] == 1 and row['exhausts'] == expected


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


@pytest.mark.parametrize('multiple_choices', [False, True])
def test_dual_wield_new_copies_reset_dynamic_exhaust(combat_factory, multiple_choices):
    env = combat_factory(['STRIKE_RED'], 'Cultist')
    bc = env.thick.bc
    hand = [native_card('DUAL_WIELD', 0), native_card('STRIKE_RED', 1, exhaust=True)]
    if multiple_choices:
        hand.append(native_card('BASH', 2, base_cost=2, cost=2))
    bc.set_card_state(native_card_state(hand=hand))
    bc.execute(sts.Action(sts.ActionType.CARD, 0))
    if multiple_choices:
        bc.execute(sts.Action(sts.ActionType.SINGLE_CARD_SELECT, 0))
    strikes = [c for c in bc.card_state()['hand'] if c['id'] == 'STRIKE_RED']
    assert len(strikes) == 2
    assert sum(c['exhausts'] for c in strikes) == (0 if multiple_choices else 1)
    assert all(not c['exhausts'] for c in strikes if c['uid'] >= 100)


def test_anger_new_copy_does_not_inherit_dynamic_exhaust(combat_factory):
    env = combat_factory(['STRIKE_RED'], 'Cultist')
    bc = env.thick.bc
    bc.set_card_state(native_card_state(hand=[native_card('ANGER', 1, cost=0, base_cost=0, exhaust=True)]))
    bc.execute(sts.Action(sts.ActionType.CARD, 0, 0))
    state = bc.card_state()
    assert [(c['uid'], c['exhausts']) for c in state['exhaust']] == [(1, True)]
    assert [(c['uid'], c['exhausts']) for c in state['discard']] == [(100, False)]


@pytest.mark.timeout(240)
@pytest.mark.parametrize('variant', range(5), ids=['hand', 'draw', 'discard', 'exhaust', 'action-dedup'])
def test_dynamic_exhaust_search_contract(game_context_executable, variant):
    import subprocess
    executable, run_env = game_context_executable
    result = subprocess.run([str(executable), 'dynamic-exhaust-search', str(variant), '20'],
                            env=run_env, capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.endswith('PASS\n')
