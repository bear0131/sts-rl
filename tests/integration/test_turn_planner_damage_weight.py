"""Attack/block preference is explicit and never overrides sampled death risk."""
import pytest
import slaythespire as sts

from env.search.turn_planner import NativeTurnPlanner, TurnPlannerConfig


@pytest.mark.parametrize('weight', [0, -1, float('inf'), float('nan')])
def test_damage_preference_requires_finite_positive_weight(weight):
    with pytest.raises(ValueError, match='damage_weight'):
        TurnPlannerConfig(damage_weight=weight)


def test_damage_preference_changes_live_tradeoff_but_preserves_survival(combat_factory):
    battle = combat_factory(['STRIKE_RED'] * 10).thick.bc
    battle.apply_custom_state({'player': {'hp': 60, 'energy': 1},
        'hand': ['STRIKE_RED', 'DEFEND_RED'], 'draw': [], 'discard': [],
        'monsters': [{'hp': 50, 'block': 0, 'move': 'JAW_WORM_CHOMP'}]})
    def choose(weight):
        plan = NativeTurnPlanner(TurnPlannerConfig(worlds=1, node_budget=128,
            damage_weight=weight)).choose(battle, state_nonce=0, public_relics=())
        action = sts.Action(plan.action_bits)
        assert action.action_type == sts.ActionType.CARD
        return battle.to_dict()['hand'][action.source_idx][0]
    assert choose(.26) == 'DEFEND_RED'
    assert choose(2.) == 'STRIKE_RED'
    battle.apply_custom_state({'player': {'hp': 7}})
    assert choose(2.) == 'DEFEND_RED'
