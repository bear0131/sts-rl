"""Shared public-relic semantics used by all optional upper policies."""
from copy import deepcopy

import pytest

from env.core import action_space as A
from rl.evaluation.baselines.overworld_rules import (
    COST_ONLY_UPGRADES, card_value, rank_overworld, relic_strength_sources, upgrade_value,
)
from rl.evaluation.baselines.overworld_engine import (
    engine_card_value, engine_profile, engine_upgrade_value, rank_engine_overworld,
)
from rl.evaluation.baselines.overworld_survival import coverage, survival_upgrade_value


@pytest.fixture
def state():
    return {'phase': 'card_select', 'options': {'cards': [('Dark Embrace', 0), ('Uppercut', 0)],
        'select': {'context': 'UPGRADE'}}, 'global': dict(act=2, floor=27, hp=50,
        max_hp=75, ascension=20, gold=100, boss='CHAMP', relics=[], relic_counters=[],
        deck=[('AscendersBane', 0), ('Strike_R', 0), ('Defend_R', 0),
              ('Dark Embrace', 0), ('Uppercut', 0)])}


@pytest.mark.parametrize('cid', sorted(COST_ONLY_UPGRADES))
def test_snecko_removes_cost_only_upgrade_value_in_all_uppers(state, cid):
    state['global']['relics'] = ['Snecko Eye']
    for value in (upgrade_value, engine_upgrade_value, survival_upgrade_value):
        assert value((cid, 0), state) == 0
    state['global']['relics'] = []
    assert upgrade_value((cid, 0), state) > 0


def test_snecko_upgrade_screen_prefers_actual_effect_and_preserves_mixed_benefits(state):
    state['global']['relics'] = ['Snecko Eye']
    legal = [A._PICK, A._PICK + 1]
    assert rank_overworld(state, legal)[0]['action'] == A._PICK + 1
    assert rank_engine_overworld(state, legal)[0][0]['action'] == A._PICK + 1
    assert upgrade_value(('Uppercut', 0), state) == 10
    assert upgrade_value(('Blood For Blood', 0), state) > 0  # damage still rises
    assert engine_upgrade_value(('Burning Pact', 0), state) > 0  # draw still rises
    # Cost-only upgraded and base versions have the same effect after Confused.
    assert engine_card_value(('Dark Embrace', 1), state) == engine_card_value(('Dark Embrace', 0), state)


@pytest.mark.parametrize('relic,counter,curse,expected', [
    ('Vajra', 0, False, 1), ('Brimstone', 0, False, 1),
    ('Girya', 0, True, 0), ('Girya', 1, False, 1), ('Girya', 3, False, 1),
    ('Du-Vu Doll', 0, False, 0), ('Du-Vu Doll', 1, True, 1),
    ('Red Skull', 0, False, 0), ('Sling of Courage', 0, False, 0),
    ('MutagenicStrength', 0, False, 0),
])
def test_strength_sources_are_present_not_the_strength_amount(state, relic, counter, curse, expected):
    state['global']['relics'] = [relic]
    state['global']['relic_counters'] = [counter]
    state['global']['deck'] = [('Strike_R', 0)] + ([('AscendersBane', 0)] if curse else [])
    before = deepcopy(state)
    assert relic_strength_sources(state) == expected
    assert engine_profile(state).strength == expected
    assert coverage(state).strength == expected
    assert state == before


def test_unknown_girya_counter_does_not_invent_strength_and_duvu_needs_real_curse(state):
    state['global'].update(relics=['Girya'], relic_counters=[])
    assert relic_strength_sources(state) == 0
    state['global'].update(relics=['Du-Vu Doll'], relic_counters=[99], deck=[('Strike_R', 0)])
    assert relic_strength_sources(state) == 0
    state['global']['deck'].append(('AscendersBane', 0))
    assert relic_strength_sources(state) == 1


def test_public_relic_strength_enables_limit_break_without_a_strength_card(state):
    state['phase'] = 'combat_reward'
    state['options'] = {'cards': [('Limit Break', 0), ('Clash', 0)]}
    state['global']['relics'], state['global']['relic_counters'] = ['Brimstone'], [0]
    legal = [A._RCARD, A._RCARD + 1, A._RSKIP]
    assert rank_engine_overworld(state, legal)[0][0]['action'] == A._RCARD
    with_source = card_value(('Limit Break', 0), state)
    state['global']['relics'] = []
    assert rank_engine_overworld(state, legal)[0][0]['action'] == A._RSKIP
    assert card_value(('Limit Break', 0), state) < with_source
