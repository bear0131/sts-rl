"""FRONT02: equivalent live plans must not postpone useful actions forever."""
import json

import pytest
import slaythespire as sts

import env.search.turn_planner as planner_module
from env.search.turn_planner import NativeTurnPlanner, TurnPlannerConfig
from rl.evaluation.baselines.turn_planner import TurnPlannerStrategy

pytestmark = pytest.mark.native
END = 4 << 29
IRON_WAVE_AT_CHOSEN = (1 << 16) | 1


def _chosen_loop(combat_factory, *, productive=False):
    env = combat_factory(['STRIKE_RED'] * 10, encounter='Cultist And Chosen',
                         relics=['Medical Kit'])
    env.thick.gc = sts.GameContext(sts.CharacterClass.IRONCLAD, 42, 20)
    env.thick.gc.set_deck([('STRIKE_RED', 0)] * 10)
    env.thick.gc.set_relics(['Medical Kit'])
    battle = sts.create_battle(env.thick.gc, 'Cultist And Chosen')
    env.thick.bc = battle
    # Binary Hex must come from the real monster action. Its trigger replaces
    # exhausted Dazed before Dark Embrace draws, even with an empty draw pile.
    battle.apply_custom_state({'monsters': [{'hp': 0}, {'move': 'CHOSEN_HEX'}]})
    battle.execute(sts.Action(END))
    powers = [('DARK_EMBRACE', 1), ('COMBUST', 5), ('STRENGTH', 1),
              ('VULNERABLE', 2), ('METALLICIZE', 3)]
    if productive:
        powers.append(('FEEL_NO_PAIN', 2))
    battle.apply_custom_state({
        'player': {'hp': 73, 'max_hp': 80, 'energy': 0 if productive else 1,
                   'gold': 124, 'powers': powers},
        'hand': ['FLAME_BARRIER', 'IRON_WAVE', 'CLOTHESLINE', 'DAZED'],
        'draw': [], 'discard': ['STRIKE_RED'] * 16, 'exhaust': [],
        'monsters': [{'hp': 0}, {'hp': 25, 'block': 0, 'move': 'CHOSEN_POKE',
                                'powers': [('VULNERABLE', 1), ('WEAK', 1)]}],
    })
    return battle


def _planner(**kwargs):
    return NativeTurnPlanner(TurnPlannerConfig(worlds=2, node_budget=2048, **kwargs))


def test_a20_real_hex_loop_prefers_equal_live_damage_and_replanning_finishes_turn(combat_factory):
    battle = _chosen_loop(combat_factory)
    before = battle.to_dict()
    loop = battle.clone()
    loop.execute(sts.Action(3))
    after = loop.to_dict()
    assert after['hand'] == before['hand'] and after['draw'] == before['draw'] == []
    assert after['monsters'][1]['hp'] == 25
    assert len(after['exhaust']) == len(before['exhaust']) + 1

    planner = _planner()
    result = planner.choose(battle, state_nonce=3978, public_relics=('Medical Kit',))
    values = result.diagnostics['root_values']
    outcomes = result.diagnostics['root_outcomes']
    assert values['3'] == values[str(IRON_WAVE_AT_CHOSEN)] > values[str(END)]
    assert all(v['wins'] == v['deaths'] == v['unresolved'] == 0 for v in outcomes.values())
    previous = max((int(bits) for bits in values), key=lambda bits: (values[str(bits)], -bits))
    assert previous == 3
    assert result.action_bits == IRON_WAVE_AT_CHOSEN
    assert outcomes[str(IRON_WAVE_AT_CHOSEN)]['completed_steps_total'] == 4
    assert outcomes['3']['completed_steps_total'] == 6
    assert battle.to_dict() == before
    json.dumps(result.diagnostics, allow_nan=False)

    battle.execute(sts.Action(result.action_bits))
    assert battle.to_dict()['monsters'][1]['hp'] == 16
    # Replan after execution, using the same incrementing public step nonce as
    # the real adapter. Progress does not rely on executing a cached sequence.
    for step in range(3979, 3987):
        result = planner.choose(battle, state_nonce=step, public_relics=('Medical Kit',))
        battle.execute(sts.Action(result.action_bits))
        if battle.turn > before['turn']:
            break
    assert battle.turn > before['turn']
    assert battle.to_dict()['monsters'][1]['hp'] < 16


def test_productive_exhaust_loop_still_builds_block_before_ending_turn(combat_factory):
    battle = _chosen_loop(combat_factory, productive=True)
    before = battle.to_dict()
    result = _planner().choose(battle, state_nonce=3978, public_relics=('Medical Kit',))
    outcomes = result.diagnostics['root_outcomes']
    assert outcomes['3']['heuristic_mean'] > outcomes[str(END)]['heuristic_mean']
    assert outcomes['3']['completed_steps_total'] > outcomes[str(END)]['completed_steps_total']
    assert result.action_bits == 3
    battle.execute(sts.Action(result.action_bits))
    assert battle.to_dict()['player']['block'] == before['player']['block'] + 2
    assert battle.to_dict()['hand'] == before['hand']


def test_unresolved_partial_does_not_gain_or_lose_a_completion_tie(combat_factory, monkeypatch):
    battle = combat_factory(['PURITY'] * 5).thick.bc
    battle.apply_custom_state({'hand': ['PURITY', 'DEFEND_RED', 'STRIKE_RED']})
    monkeypatch.setattr(planner_module, 'state_value', lambda raw, **kwargs: 1.0)
    result = NativeTurnPlanner(TurnPlannerConfig(worlds=1, node_budget=1,
        beam_width=1, max_depth=1)).choose(battle, state_nonce=3, public_relics=())
    outcomes = result.diagnostics['root_outcomes']
    assert outcomes['0']['unresolved'] == 1 and outcomes['0']['completed_steps_total'] == 0
    assert outcomes[str(END)]['boundary_live'] == 1
    assert len({v['heuristic_mean'] for v in outcomes.values()}) == 1
    assert result.action_bits == 0  # old stable action ordering; no false proof


def test_unavoidable_death_does_not_acquire_shortest_death_preference(combat_factory):
    battle = combat_factory(['FLEX'] * 5).thick.bc
    battle.apply_custom_state({'player': {'hp': 1, 'energy': 0}, 'hand': ['FLEX'],
                               'monsters': [{'hp': 100, 'move': 'JAW_WORM_CHOMP'}]})
    result = _planner().choose(battle, state_nonce=3, public_relics=())
    assert all(v['deaths'] == 2 for v in result.diagnostics['root_outcomes'].values())
    assert result.action_bits == 0


def test_manifest_declares_restricted_completed_tie():
    assert TurnPlannerStrategy().manifest()['completed_tie_break'] == (
        'fewest-actions-if-all-tied-roots-fully-resolved-and-surviving')
