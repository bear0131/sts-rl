"""Preserve a damage channel without rejecting known non-Attack engines."""
import pytest
import slaythespire as sts

from env.search.turn_planner import (
    NativeTurnPlanner, TurnPlannerConfig, no_remaining_damage_source, state_value,
)
from rl.evaluation.baselines.turn_planner import TurnPlannerStrategy
from rl.evaluation.strategies import DecisionRequest


def _public_relics(env):
    return tuple(relic['id'] for relic in env.thick.gc.relic_state())


def _configure(env, *, hand, draw=(), discard=(), powers=(), hp=40):
    env.thick.bc.apply_custom_state({
        'player': {'hp': 80, 'max_hp': 80, 'energy': 3, 'powers': list(powers)},
        'hand': list(hand), 'draw': list(draw), 'discard': list(discard), 'exhaust': [],
        'monsters': [{'hp': hp, 'block': 0, 'move': 'JAW_WORM_CHOMP'}],
    })
    env._invalidate_backend_cache()
    return env.thick.bc


def _card(battle, name):
    index = next(i for i, c in enumerate(battle.to_dict()['hand']) if c[0] == name)
    return next(a for a in battle.rule_actions()
                if a.action_type == sts.ActionType.CARD and a.source_idx == index)


def _fiend_fire_fixture(combat_factory):
    env = combat_factory(['FIEND_FIRE'] * 5, encounter='Champ')
    battle = _configure(env, hand=['FIEND_FIRE', 'STRIKE_RED', 'WOUND'],
                        draw=['DEFEND_RED'], hp=100)
    battle.apply_custom_state({'monsters': [{'move': 'THE_CHAMP_GLOAT'}]})
    return env


def test_native_fiend_fire_preserves_last_repeatable_attack(combat_factory):
    env = _fiend_fire_fixture(combat_factory)
    battle = env.thick.bc
    planner = NativeTurnPlanner(TurnPlannerConfig(worlds=2, node_budget=512))
    # The old preference is observable using no relic context: FF gains one
    # immediate damage over Strike -> FF, but consumes the only reusable attack.
    old = planner.choose(battle, state_nonce=888)
    assert battle.to_dict()['hand'][sts.Action(old.action_bits).source_idx][0] == 'FIEND_FIRE'
    result = planner.choose(battle, state_nonce=888, public_relics=_public_relics(env))
    chosen = sts.Action(result.action_bits)
    assert battle.to_dict()['hand'][chosen.source_idx][0] == 'STRIKE_RED'

    doomed = battle.clone()
    doomed.execute(_card(doomed, 'FIEND_FIRE'))
    lost_resources = doomed.preview_end_turn_result(4 << 29)
    assert lost_resources['monsters'][0]['hp'] == 86
    assert no_remaining_damage_source(lost_resources, ())
    assert [c[0] for c in lost_resources['draw'] + lost_resources['discard']] == ['DEFEND_RED']

    saved = battle.clone()
    saved.execute(chosen)
    saved.execute(_card(saved, 'FIEND_FIRE'))
    retained = saved.preview_end_turn_result(4 << 29)
    assert retained['monsters'][0]['hp'] == 87
    assert not no_remaining_damage_source(retained, ())
    assert any(c[0] == 'STRIKE_RED' for c in retained['discard'])
    assert state_value(retained, public_relics=()) > state_value(lost_resources, public_relics=())


def test_adapter_supplies_public_relics_to_resource_guard(combat_factory):
    env = _fiend_fire_fixture(combat_factory)
    env.combat_action_contract = 'native-v1'
    request = DecisionRequest(env, env._obs(), env._info(), 42, 888)
    result = TurnPlannerStrategy().decide([request])[0]
    assert result.native_action is not None
    assert env.thick.bc.to_dict()['hand'][sts.Action(result.native_action).source_idx][0] == 'STRIKE_RED'


def test_immediate_native_victory_is_not_penalized_for_exhausting_attacks(combat_factory):
    env = _fiend_fire_fixture(combat_factory)
    env.thick.bc.apply_custom_state({'monsters': [{'hp': 14}]})
    result = NativeTurnPlanner().choose(env.thick.bc, state_nonce=888, public_relics=())
    env.thick.bc.execute(sts.Action(result.action_bits))
    assert int(env.thick.bc.outcome) == 1
    assert state_value(env.thick.bc.to_dict(), public_relics=()) > 1000


@pytest.mark.parametrize('hand,powers,sequence,expected_damage', [
    (['DEFEND_RED', 'BODY_SLAM'], [], ['DEFEND_RED', 'BODY_SLAM'], 5),
    (['DEFEND_RED'], [('JUGGERNAUT', 5)], ['DEFEND_RED'], 5),
    (['DEFEND_RED'], [('COMBUST', 5)], [], 5),
    (['FLAME_BARRIER'], [], ['FLAME_BARRIER'], 4),
])
def test_nonstandard_native_damage_sources_remain_possible(
        combat_factory, hand, powers, sequence, expected_damage):
    env = combat_factory(['STRIKE_RED'] * 5)
    battle = _configure(env, hand=hand, powers=powers)
    assert not no_remaining_damage_source(battle.to_dict(), ())
    before = battle.to_dict()['monsters'][0]['hp']
    for card in sequence:
        battle.execute(_card(battle, card))
    if not sequence or 'FLAME_BARRIER' in sequence:
        battle.execute(sts.Action(sts.ActionType.END_TURN))
    assert before - battle.to_dict()['monsters'][0]['hp'] == expected_damage


@pytest.mark.parametrize('relic,hand,sequence,expected_damage', [
    ('Mercury Hourglass', ['DEFEND_RED'], [], 3),
    ('Bronze Scales', ['DEFEND_RED'], [], 3),
    ("Charon's Ashes", [('TRUE_GRIT', 1), 'WOUND'], ['TRUE_GRIT'], 3),
    ('Letter Opener', ['DEFEND_RED'] * 3, ['DEFEND_RED'] * 3, 5),
])
def test_native_damage_relics_do_not_become_false_resource_dead_ends(
        combat_factory, relic, hand, sequence, expected_damage):
    env = combat_factory(['STRIKE_RED'] * 5, relics=[relic])
    battle = _configure(env, hand=hand)
    assert not no_remaining_damage_source(battle.to_dict(), _public_relics(env))
    before = battle.to_dict()['monsters'][0]['hp']
    for card in sequence:
        battle.execute(_card(battle, card))
    if not sequence:
        battle.execute(sts.Action(sts.ActionType.END_TURN))
    assert before - battle.to_dict()['monsters'][0]['hp'] == expected_damage


def test_delayed_bomb_and_status_damage_are_not_mistaken_for_no_output(combat_factory):
    env = combat_factory(['STRIKE_RED'] * 5)
    battle = _configure(env, hand=['THE_BOMB', 'DEFEND_RED'])
    battle.execute(_card(battle, 'THE_BOMB'))
    assert any(battle.to_dict()['player']['combat_counters'][f'bomb_{n}'] for n in (1, 2, 3))
    assert not no_remaining_damage_source(battle.to_dict(), ())

    battle = _configure(env, hand=['DEFEND_RED'], draw=['WOUND'], powers=[('FIRE_BREATHING', 6)])
    assert not no_remaining_damage_source(battle.to_dict(), ())
    before = battle.to_dict()['monsters'][0]['hp']
    battle.execute(sts.Action(sts.ActionType.END_TURN))
    assert before - battle.to_dict()['monsters'][0]['hp'] == 6


def test_unknown_context_generation_and_survival_encounters_are_conservative(combat_factory):
    env = combat_factory(['STRIKE_RED'] * 5)
    battle = _configure(env, hand=['DEFEND_RED'])
    assert no_remaining_damage_source(battle.to_dict(), ())
    assert not no_remaining_damage_source(battle.to_dict())  # Relics unavailable.
    for card in ('INFERNAL_BLADE', 'EXHUME', 'DISCOVERY', 'COMBUST'):
        battle = _configure(env, hand=[card, 'DEFEND_RED'])
        assert not no_remaining_damage_source(battle.to_dict(), ())
    for relic in ('Dead Branch', "Nilry's Codex", 'Stone Calendar'):
        battle = _configure(env, hand=['DEFEND_RED'])
        assert not no_remaining_damage_source(battle.to_dict(), (relic,))
    transient = combat_factory(['DEFEND_RED'] * 5, encounter='Transient').thick.bc
    assert not no_remaining_damage_source(transient.to_dict(), ())


def test_pending_exhume_can_recover_the_only_attack(combat_factory):
    env = combat_factory(['STRIKE_RED'] * 5)
    battle = _configure(env, hand=['EXHUME', 'DEFEND_RED'])
    battle.apply_custom_state({'exhaust': ['STRIKE_RED', 'DEFEND_RED']})
    battle.execute(_card(battle, 'EXHUME'))
    assert battle.to_dict()['input_state'] == 2
    assert not no_remaining_damage_source(battle.to_dict(), ())
    battle.execute(next(a for a in battle.rule_actions()
                        if a.action_type == sts.ActionType.SINGLE_CARD_SELECT and a.source_idx == 0))
    battle.execute(_card(battle, 'STRIKE_RED'))
    assert battle.to_dict()['monsters'][0]['hp'] == 34
