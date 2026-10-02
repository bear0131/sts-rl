"""Time Eater's public clock informs planner leaves."""
import pytest
import slaythespire as sts

from env.search.turn_planner import (
    LEGACY_LEAF_PROFILE,
    NativeTurnPlanner,
    TIME_CLOCK_LEAF_PROFILE,
    TurnPlannerConfig,
    state_value,
)


def test_time_eater_boundary_prices_the_public_remaining_card_budget(combat_factory):
    battle = combat_factory(['POMMEL_STRIKE'] * 25, encounter='Time Eater').thick.bc
    battle.apply_custom_state({
        'player': {'energy': 20},
        'hand': ['POMMEL_STRIKE'] * 5,
        'draw': ['POMMEL_STRIKE'] * 20,
        'discard': [],
    })
    for _ in range(11):
        battle.execute(sts.Action(sts.ActionType.CARD, 0, 0))
    raw = battle.execute_to_enemy_boundary(4 << 29)
    assert dict(raw['monsters'][0]['powers'])['Time Warp'] == 11
    legacy = state_value(raw, horizon=3, leaf_profile=LEGACY_LEAF_PROFILE)
    candidate = state_value(raw, horizon=3, leaf_profile=TIME_CLOCK_LEAF_PROFILE)
    assert candidate == pytest.approx(legacy - 48)


def test_non_time_eater_keeps_legacy_values_and_root_choice(combat_factory):
    battle = combat_factory(['STRIKE_RED', 'DEFEND_RED'] * 5).thick.bc
    battle.apply_custom_state({
        'hand': ['STRIKE_RED', 'DEFEND_RED'],
        'draw': ['STRIKE_RED', 'DEFEND_RED'] * 4,
        'discard': [],
        'exhaust': [],
    })
    raw = battle.to_dict()
    assert state_value(raw, horizon=3, leaf_profile=TIME_CLOCK_LEAF_PROFILE) == (
        state_value(raw, horizon=3, leaf_profile=LEGACY_LEAF_PROFILE))
    common = dict(worlds=2, node_budget=128, beam_width=8, max_depth=16,
                  master_nonce=20260918, lookahead_turns=1)
    old = NativeTurnPlanner(TurnPlannerConfig(
        **common, leaf_profile=LEGACY_LEAF_PROFILE)).choose(
            battle, state_nonce=55, public_relics=())
    new = NativeTurnPlanner(TurnPlannerConfig(
        **common, leaf_profile=TIME_CLOCK_LEAF_PROFILE)).choose(
            battle, state_nonce=55, public_relics=())
    assert new.action_bits == old.action_bits
    assert new.diagnostics['root_values'] == old.diagnostics['root_values']
