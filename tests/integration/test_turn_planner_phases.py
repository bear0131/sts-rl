"""Phase progress is continuous across native slime splits, not raw HP gain."""
import pytest
import slaythespire as sts

from env.search.turn_planner import NativeTurnPlanner, TurnPlannerConfig, enemy_burden, state_value


def _burden(battle):
    return enemy_burden(battle.to_dict()['monsters'])


@pytest.mark.parametrize('encounter,max_hp,start_hp', [
    ('Large Slime', 65, 33), ('Large Slime', 66, 34),
    ('Slime Boss', 140, 71), ('Slime Boss', 141, 72),
])
def test_native_split_preserves_remaining_phase_burden(
        combat_factory, encounter, max_hp, start_hp):
    battle = combat_factory(['STRIKE_RED'] * 5, encounter=encounter).thick.bc
    battle.apply_custom_state({'player': {'hp': 80, 'max_hp': 80, 'energy': 3},
        'hand': ['STRIKE_RED'] * 5, 'draw': [], 'discard': [],
        'monsters': [{'hp': start_hp, 'max_hp': max_hp, 'block': 0, 'powers': []}]})
    initial_burden = _burden(battle)
    battle.execute(sts.Action(sts.ActionType.CARD, 0, 0))
    before = battle.to_dict()
    assert before['monsters'][0]['move'].endswith('_SPLIT')
    pending_burden = _burden(battle)
    after = battle.execute_to_enemy_boundary(4 << 29)
    assert enemy_burden(after['monsters']) == pending_burden
    assert pending_burden[0] < initial_burden[0]
    assert pending_burden[1] == initial_burden[1]
    # No block, Strength or debuff is removed in this fixture. Merely turning
    # one parent into children must not create a spurious leaf-score penalty.
    assert state_value(after, horizon=3) == pytest.approx(state_value(before, horizon=3))


def test_boss_children_second_split_keeps_other_live_monster(combat_factory):
    battle = combat_factory(['STRIKE_RED'] * 5, encounter='Slime Boss').thick.bc
    battle.apply_custom_state({'hand': ['STRIKE_RED'] * 5,
        'monsters': [{'hp': 71, 'max_hp': 140, 'block': 0, 'powers': []}]})
    battle.execute(sts.Action(sts.ActionType.CARD, 0, 0))
    battle.execute(sts.Action(sts.ActionType.END_TURN))
    raw = battle.to_dict()
    # The native boss corpse occupies slot1; do not compact or retarget slots.
    assert [m['id'] for m in raw['monsters']] == ['SPIKE_SLIME_L', 'SLIME_BOSS', 'ACID_SLIME_L']
    battle.apply_custom_state({'player': {'energy': 3}, 'hand': ['STRIKE_RED'] * 5,
        'monsters': [{'hp': 33, 'block': 0, 'powers': []}, {}, {'move': 'ACID_SLIME_L_LICK'}]})
    battle.execute(sts.Action(sts.ActionType.CARD, 0, 0))
    before = _burden(battle)
    after = battle.execute_to_enemy_boundary(4 << 29)
    assert enemy_burden(after['monsters']) == before
    assert sum(m['hp'] > 0 and not m['gone'] for m in after['monsters']) == 3
    assert any(m['id'] == 'ACID_SLIME_L' and m['hp'] == 65 for m in after['monsters'])


def test_ordinary_monsters_and_kills_keep_the_original_scale(combat_factory):
    battle = combat_factory(['STRIKE_RED'] * 5).thick.bc
    battle.apply_custom_state({'hand': ['STRIKE_RED'] * 5,
                              'monsters': [{'hp': 6, 'powers': []}]})
    assert _burden(battle) == (6, 1)
    battle.execute(sts.Action(sts.ActionType.CARD, 0, 0))
    assert _burden(battle) == (0, 0)
    assert int(battle.outcome) == 1


def test_metallicize_disarm_slime_fixture_uses_available_attacks(combat_factory):
    # Reduced public mechanism from R2 seed150016, floor7, turn8. This does not
    # reconstruct the run's private RNG or claim its eventual counterfactual win.
    battle = combat_factory(['STRIKE_RED'] * 5, encounter='Large Slime').thick.bc
    battle.apply_custom_state({'turn': 8, 'player': {
        'hp': 77, 'max_hp': 80, 'energy': 4, 'powers': [('METALLICIZE', 10)]},
        'hand': [('DISARM', 1)], 'draw': [], 'discard': [], 'exhaust': [],
        'monsters': [{'hp': 35, 'max_hp': 65, 'block': 0,
            'move': 'ACID_SLIME_L_TACKLE', 'powers': [('VULNERABLE', 1)]}]})
    battle.execute(sts.Action(sts.ActionType.CARD, 0, 0))
    battle.apply_custom_state({'hand': ['STRIKE_RED'] * 5})
    assert dict(battle.to_dict()['monsters'][0]['powers'])['Strength'] == -3
    result = NativeTurnPlanner(TurnPlannerConfig(worlds=2, node_budget=512)).choose(
        battle, state_nonce=8, public_relics=())
    assert sts.Action(result.action_bits).action_type == sts.ActionType.CARD
    assert result.diagnostics['root_values'][str(result.action_bits)] > result.diagnostics['root_values'][str(4 << 29)]
