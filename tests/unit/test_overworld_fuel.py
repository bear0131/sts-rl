"""Fuel profiles distinguish useful conversion inputs from generic deck bloat."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from env.core import action_space as A
from rl.evaluation.baselines.overworld_engine import (
    EngineUpperStrategy, FuelUpperStrategy, engine_card_value, engine_profile,
    engine_relic_value, engine_skip_threshold, fuel_card_value, fuel_relic_value,
    rank_engine_overworld,
)
from rl.evaluation.strategies import DecisionRequest, strategy_manifest


@pytest.fixture
def state():
    return {'phase': 'combat_reward', 'options': {'cards': [('Shrug It Off', 0)]},
        'global': dict(act=3, ascension=20, floor=45, boss='DONU_AND_DECA',
            hp=50, max_hp=80, gold=100, relics=[], relic_counters=[],
            deck=[('Strike_R', 0)] * 5 + [('Defend_R', 0)] * 7 +
                [('AscendersBane', 0), ('Corruption', 0), ('Feel No Pain', 0),
                 ('Feel No Pain', 0), ('Dark Embrace', 0), ('Carnage', 0),
                 ('Anger', 0), ('Pommel Strike', 1), ('Offering', 0)] +
                [('Shrug It Off', 0)] * 3 + [('Disarm', 0), ('Shockwave', 0),
                 ('Spot Weakness', 0), ('Burning Pact', 0), ('Power Through', 0),
                 ('Fiend Fire', 0)])}


def test_supported_engine_accepts_useful_fuel_and_stack_control_rejects_it(state):
    threshold = engine_skip_threshold(engine_profile(state))
    assert engine_card_value(('Shrug It Off', 0), state) < threshold
    assert fuel_card_value(('Shrug It Off', 0), state) > threshold
    assert engine_card_value(('Feel No Pain', 0), state) < threshold
    assert fuel_card_value(('Feel No Pain', 0), state) > threshold
    legal = [A._RCARD, A._RSKIP]
    assert rank_engine_overworld(state, legal)[0][0]['action'] == A._RSKIP
    assert rank_engine_overworld(state, legal, card_value_fn=fuel_card_value)[0][0]['action'] == A._RCARD


@pytest.mark.parametrize('card', ['Cleave', 'Rampage', 'Clash', 'Flex', 'Havoc', 'Entrench', 'Infernal Blade'])
def test_engine_does_not_cancel_dilution_for_attack_or_speculative_skill(state, card):
    assert fuel_card_value((card, 0), state) == engine_card_value((card, 0), state)


def test_exhaust_fuel_requires_both_converter_and_setup_access(state):
    no_corruption = deepcopy(state)
    no_corruption['global']['deck'] = [c for c in state['global']['deck'] if c[0] != 'Corruption']
    assert fuel_card_value(('Shrug It Off', 0), no_corruption) == engine_card_value(('Shrug It Off', 0), no_corruption)
    no_draw = deepcopy(state)
    no_draw['global']['deck'] = [c for c in state['global']['deck'] if c[0] not in (
        'Dark Embrace', 'Offering', 'Pommel Strike', 'Burning Pact', 'Shrug It Off')]
    assert fuel_card_value(('Shrug It Off', 0), no_draw) == engine_card_value(('Shrug It Off', 0), no_draw)
    no_draw['global']['relics'] = ['Snecko Eye']
    assert fuel_card_value(('Shrug It Off', 0), no_draw) > engine_card_value(('Shrug It Off', 0), no_draw)


def test_no_engine_and_opening_damage_starvation_keep_compact_values(state):
    state['global'].update(act=1, floor=3, relics=['Mummified Hand', 'Bird Faced Urn'])
    state['global']['deck'] = [('Strike_R', 0)] * 5 + [('Defend_R', 0)] * 4 + [
        ('Corruption', 0), ('Dark Embrace', 0), ('Feel No Pain', 0)]
    for card in ('Carnage', 'Shrug It Off', 'Feel No Pain', 'Dark Embrace', 'Demon Form'):
        assert fuel_card_value((card, 0), state) == engine_card_value((card, 0), state)


def test_duplicate_skill_bonus_cannot_support_unbounded_fuel_collection(state):
    before = fuel_card_value(('Shrug It Off', 0), state)
    state['global']['deck'] += [('Shrug It Off', 0)] * 5
    assert fuel_card_value(('Shrug It Off', 0), state) < before - 5
    assert fuel_card_value(('Shrug It Off', 0), state) < engine_skip_threshold(engine_profile(state))


def test_power_relics_use_native_and_canonical_ids_without_enabling_dead_payoffs(state):
    ordinary = fuel_card_value(('Feel No Pain', 0), state)
    for ids in (['Mummified Hand', 'Bird Faced Urn'], ['MUMMIFIED_HAND', 'BIRD_FACED_URN']):
        state['global']['relics'] = ids
        assert fuel_card_value(('Feel No Pain', 0), state) > ordinary + 2
        assert fuel_card_value(('Rupture', 0), state) == engine_card_value(('Rupture', 0), state)
        assert fuel_relic_value('Mummified Hand', state) > engine_relic_value('Mummified Hand', state)
    state['global']['relics'] = ["Charon's Ashes"]
    assert fuel_card_value(('Shrug It Off', 0), state) > engine_card_value(('Shrug It Off', 0), state)


def test_fuel_removal_preserves_supported_skill_value(state):
    state['phase'] = 'card_select'
    state['options'] = {'cards': [('Shrug It Off', 0), ('Clash', 0)], 'select': {'context': 'REMOVE'}}
    rows = rank_engine_overworld(state, [A._PICK, A._PICK + 1], card_value_fn=fuel_card_value)[0]
    assert rows[0]['action'] == A._PICK + 1


def test_named_strategy_reuses_batching_and_exposes_profile_and_trace(state):
    mask = np.zeros(A.N_ACTIONS, dtype=bool)
    mask[[A._RCARD, A._RSKIP]] = True
    thick = SimpleNamespace(obs_canonical=lambda: state, rmap=None, gc=SimpleNamespace(cur_map_node_y=0))
    env = SimpleNamespace(thick=thick, action_masks=lambda: mask)
    env.unwrapped = env
    request = DecisionRequest(env, {}, {'phase': 'combat_reward'}, 901, 0)
    before, trace = deepcopy(state), []
    fuel = FuelUpperStrategy(trace=trace.append)
    assert EngineUpperStrategy().decide([request])[0].action == A._RSKIP
    decision = fuel.decide([request])[0]
    assert decision.action == A._RCARD
    assert decision.source == 'overworld-fuel'
    assert decision.diagnostics['overworld_fuel_decisions'] == 1
    assert strategy_manifest(fuel)['draft_profile'] == 'fuel'
    assert trace[0]['draft_profile'] == 'fuel'
    assert trace[0]['version'] == fuel.version
    assert state == before
    assert fuel.decide([]) == []
