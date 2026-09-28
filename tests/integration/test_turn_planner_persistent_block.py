"""Removing enemy Barricade block remains progress above the old30-block cap."""
import pytest
import slaythespire as sts

from env.search.turn_planner import NativeTurnPlanner, TurnPlannerConfig, state_value

pytestmark = pytest.mark.native
END = 4 << 29


def _shielded(combat_factory, encounter, block):
    battle = combat_factory(['STRIKE_RED'] * 5, encounter=encounter).thick.bc
    battle.apply_custom_state({
        'player': {'hp': 50, 'block': 50, 'energy': 1},
        'hand': ['STRIKE_RED'], 'draw': ['DEFEND_RED'] * 5, 'discard': [],
        'monsters': [{'block': block}],
    })
    return battle


@pytest.mark.parametrize('block', [45, 75, 166])
def test_free_attack_reduces_persistent_burden_even_without_hp_damage(combat_factory, block):
    battle = _shielded(combat_factory, 'Spheric Guardian', block)
    before = battle.to_dict()
    assert dict(before['monsters'][0]['powers'])['Barricade'] == 1
    waiting = battle.clone()
    waiting_raw = waiting.execute_to_enemy_boundary(END)
    attacking = battle.clone()
    attacking.execute_to_enemy_boundary(0)
    attacked_raw = attacking.execute_to_enemy_boundary(END)
    assert attacked_raw['player']['hp'] == waiting_raw['player']['hp'] == 50
    assert attacked_raw['monsters'][0]['hp'] == waiting_raw['monsters'][0]['hp'] == 20
    assert waiting_raw['monsters'][0]['block'] - attacked_raw['monsters'][0]['block'] == 6
    assert attacked_raw['monsters'][0]['block'] > 30
    assert state_value(attacked_raw, horizon=1) > state_value(waiting_raw, horizon=1)

    result = NativeTurnPlanner(TurnPlannerConfig(worlds=2, node_budget=128)).choose(
        battle, state_nonce=13, public_relics=())
    assert result.action_bits == 0
    assert result.diagnostics['root_values']['0'] > result.diagnostics['root_values'][str(END)]
    assert battle.to_dict() == before


def test_ordinary_temporary_block_retains_previous_cap(combat_factory):
    battle = _shielded(combat_factory, 'Jaw Worm', 75)
    before = battle.to_dict()
    battle.execute(sts.Action(0))
    after = battle.to_dict()
    assert after['monsters'][0]['block'] == 69
    assert state_value(after, horizon=1) == state_value(before, horizon=1)
    # Temporary block really disappears at the next enemy turn; giving its
    # entire old value a long-lived burden would price useless attacks too high.
    after_turn = battle.execute_to_enemy_boundary(END)
    assert after_turn['monsters'][0]['block'] < 30
