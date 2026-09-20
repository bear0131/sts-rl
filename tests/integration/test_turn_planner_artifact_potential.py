"""Artifact stripping has value only with a remaining usable debuff source."""
import pytest
import slaythespire as sts

from env.search.turn_planner import NativeTurnPlanner, TurnPlannerConfig

pytestmark = pytest.mark.native
END = 4 << 29


def _planner():
    return NativeTurnPlanner(TurnPlannerConfig(worlds=2, node_budget=512))


def _spheric(combat_factory, *, future_debuff, relics=()):
    env = combat_factory(['STRIKE_RED'] * 5, encounter='Spheric Guardian', relics=relics)
    env.thick.gc = sts.GameContext(sts.CharacterClass.IRONCLAD, 42, 20)
    env.thick.gc.set_deck([('STRIKE_RED', 0)] * 5)
    env.thick.gc.set_relics(list(relics))
    battle = sts.create_battle(env.thick.gc, 'Spheric Guardian')
    env.thick.bc = battle
    battle.apply_custom_state({
        'player': {'energy': 0}, 'hand': ['INTIMIDATE'],
        'draw': ['BASH' if future_debuff else 'STRIKE_RED'] + ['STRIKE_RED'] * 4,
        'discard': [], 'exhaust': [],
    })
    return battle


def test_free_intimidate_strips_artifact_before_reusable_future_bash(combat_factory):
    battle = _spheric(combat_factory, future_debuff=True)
    before = battle.to_dict()
    result = _planner().choose(battle, state_nonce=1, public_relics=())
    assert result.action_bits == 0
    assert result.diagnostics['root_values']['0'] > result.diagnostics['root_values'][str(END)]
    assert result.diagnostics['root_artifact_potential']['enabled']
    assert battle.to_dict() == before
    battle.execute(sts.Action(result.action_bits))
    assert dict(before['monsters'][0]['powers'])['Artifact'] == 3
    assert dict(battle.to_dict()['monsters'][0]['powers'])['Artifact'] == 2
    assert 'Weakened' not in dict(battle.to_dict()['monsters'][0]['powers'])


def test_last_exhausting_debuff_does_not_invent_future_artifact_value(combat_factory):
    battle = _spheric(combat_factory, future_debuff=False)
    result = _planner().choose(battle, state_nonce=1, public_relics=())
    assert not result.diagnostics['root_artifact_potential']['enabled']
    assert result.diagnostics['root_values']['0'] == result.diagnostics['root_values'][str(END)]
    assert result.action_bits == END


def test_artifact_preparation_does_not_spend_last_choker_play_and_die(combat_factory):
    battle = _spheric(combat_factory, future_debuff=True, relics=('Velvet Choker',))
    battle.apply_custom_state({'hand': ['FLEX'] * 5})
    for _ in range(5):
        battle.execute(sts.Action(0))
    assert battle.to_dict()['player']['combat_counters']['cards_played'] == 5
    battle.apply_custom_state({
        'player': {'hp': 1, 'energy': 1, 'powers': [('DEXTERITY', 6)]},
        'hand': ['INTIMIDATE', 'DEFEND_RED'],
        'monsters': [{'move': 'SPHERIC_GUARDIAN_ATTACK_DEBUFF'}],
    })
    result = _planner().choose(battle, state_nonce=2, public_relics=('Velvet Choker',))
    outcomes = result.diagnostics['root_outcomes']
    assert result.diagnostics['root_artifact_potential']['enabled']
    assert outcomes['0']['deaths'] == 2
    assert outcomes['1']['deaths'] == 0
    assert result.action_bits == 1


def test_exhausted_bash_is_not_assumed_recoverable(combat_factory):
    battle = _spheric(combat_factory, future_debuff=False)
    battle.apply_custom_state({'exhaust': ['BASH']})
    result = _planner().choose(battle, state_nonce=1, public_relics=())
    assert not result.diagnostics['root_artifact_potential']['enabled']
    assert result.action_bits == END
