"""A20 Angry's unavoidable reactive Strength must not reward indefinite delay."""
import pytest
import slaythespire as sts

from env.search.turn_planner import NativeTurnPlanner, TurnPlannerConfig, state_value

pytestmark = pytest.mark.native
END = 4 << 29


def _mad_gremlin(combat_factory, *, hp=37, strength=0):
    env = combat_factory(['STRIKE_RED'] * 5, encounter='Gremlin Gang', seed=0)
    env.thick.gc = sts.GameContext(sts.CharacterClass.IRONCLAD, 35, 20)
    env.thick.gc.set_deck([('STRIKE_RED', 0)] * 5 + [('DEFEND_RED', 0)] * 4 + [('BASH', 0)])
    battle = sts.create_battle(env.thick.gc, 'Gremlin Gang')
    env.thick.bc = battle
    assert battle.to_dict()['monsters'][2]['id'] == 'MAD_GREMLIN'
    battle.apply_custom_state({
        'player': {'hp': hp, 'max_hp': 75, 'energy': 3, 'powers': []},
        'hand': ['STRIKE_RED'] * 2 + ['DEFEND_RED'] * 3,
        'draw': [], 'discard': ['DEFEND_RED', 'BATTLE_TRANCE', 'BASH', 'PERFECTED_STRIKE',
                               'STRIKE_RED', 'STRIKE_RED', 'STRIKE_RED', 'WHIRLWIND'],
        'exhaust': ['ASCENDERS_BANE'],
        'monsters': [{'hp': 0}, {'hp': 0},
                     {'hp': 25, 'block': 0, 'powers': [('STRENGTH', strength)]}, {'hp': 0}],
    })
    return battle


def _planner():
    return NativeTurnPlanner(TurnPlannerConfig(worlds=2, node_budget=2048))


def _play_to_next_turn(battle, planner, nonce=100):
    starting_turn = battle.turn
    actions = []
    for step in range(nonce, nonce + 10):
        result = planner.choose(battle, state_nonce=step, public_relics=())
        actions.append(result.action_bits)
        battle.execute(sts.Action(result.action_bits))
        if battle.is_battle_over or battle.turn > starting_turn:
            return actions
    pytest.fail('simple reduced hand should reach the next turn within ten actions')


def test_safe_damage_to_angry_is_progress_even_after_strength_trigger(combat_factory):
    battle = _mad_gremlin(combat_factory)
    assert dict(battle.to_dict()['monsters'][2]['powers'])['Angry'] == 2
    waiting = battle.clone()
    waiting.execute_to_enemy_boundary(2)  # one Defend blocks the unprovoked attack
    waiting_raw = waiting.execute_to_enemy_boundary(END)
    attacking = battle.clone()
    attacking.execute_to_enemy_boundary(2)
    attacking.execute_to_enemy_boundary(2)  # second Defend after hand compaction
    attacking.execute_to_enemy_boundary(2 << 16)  # Strike at original slot2
    attacking_raw = attacking.execute_to_enemy_boundary(END)
    assert waiting_raw['player']['hp'] == attacking_raw['player']['hp'] == 37
    assert attacking_raw['monsters'][2]['hp'] == 19
    assert dict(attacking_raw['monsters'][2]['powers'])['Strength'] == 2
    assert state_value(attacking_raw, horizon=1) > state_value(waiting_raw, horizon=1)

    planner = _planner()
    _play_to_next_turn(battle, planner)
    assert battle.to_dict()['player']['hp'] == 37
    assert battle.to_dict()['monsters'][2]['hp'] == 19
    # Repeat actual decisions and shuffled real draws; this is not a forced
    # action script and does not commit to the original search continuation.
    for nonce in (110, 120, 130, 140):
        if battle.is_battle_over:
            break
        _play_to_next_turn(battle, planner, nonce)
    assert battle.to_dict()['outcome'] == 1


def test_angry_progress_never_overrides_a_sampled_lethal_counterattack(combat_factory):
    battle = _mad_gremlin(combat_factory, hp=1, strength=10)
    before = battle.to_dict()
    planner = _planner()
    result = planner.choose(battle, state_nonce=100, public_relics=())
    attack = result.diagnostics['root_outcomes'][str(2 << 16)]
    assert attack['deaths'] == 2
    assert result.diagnostics['root_outcomes'][str(result.action_bits)]['deaths'] == 0
    _play_to_next_turn(battle, planner)
    after = battle.to_dict()
    assert after['player']['hp'] == before['player']['hp'] == 1
    assert after['monsters'][2]['hp'] == before['monsters'][2]['hp'] == 25
