"""Behavioral contracts for the explicitly limited hand-coverage experiment."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from env.core import action_space as A
from rl.evaluation.baselines.overworld_demand import (
    EnemyDemandUpperStrategy, demand_profile, hand_block_coverage,
    rank_demand_rewards, reward_adjustment,
)
from rl.evaluation.baselines.overworld_rules import rank_overworld
from rl.evaluation.strategies import Decision, DecisionRequest, strategy_manifest


@pytest.fixture
def public_state():
    return {'phase': 'combat_reward', 'options': {'cards': [('Ghostly Armor', 0), ('Battle Trance', 0)]},
        'global': dict(act=2, ascension=0, boss='AUTOMATON', floor=23, hp=70,
            max_hp=80, gold=100, relics=[], potions=['EMPTY_POTION_SLOT'] * 3,
            deck=[('Strike_R', 0)] * 18 + [('Defend_R', 0)] * 2)}


def test_uniform_hand_probability_is_hand_calculable():
    # Exactly five of six equally likely omitted-card cases contain the block.
    deck = (('Shrug It Off', 0),) + (('Strike_R', 0),) * 5
    assert hand_block_coverage(deck, 1, 99) == pytest.approx(8 * 5 / 6)
    # Draw-one is deliberately not simulated by this five-card feature.
    assert hand_block_coverage(deck, 0, 99) == 0


def test_energy_competition_is_not_sum_of_visible_block():
    deck = (('Flame Barrier', 0),) * 2 + (('Strike_R', 0),) * 4
    # Any five-card set has a Flame Barrier; two energy plays only one.
    assert hand_block_coverage(deck, 2, 99) == 12
    # Four of six hands have both copies; two have only one.
    assert hand_block_coverage(deck, 4, 99) == 20


def test_mixed_cost_subset_and_upgrade_use_real_numbers():
    deck = (('Flame Barrier', 0), ('Defend_R', 0), ('Defend_R', 0)) + (('Strike_R', 0),) * 3
    assert hand_block_coverage(deck, 2, 99) == pytest.approx((5 * 12 + 10) / 6)
    assert hand_block_coverage(deck, 3, 99) == pytest.approx((5 * 17 + 10) / 6)
    upgraded = (('Flame Barrier', 1), ('Flame Barrier', 0)) + (('Strike_R', 0),) * 4
    assert hand_block_coverage(upgraded, 2, 99) == pytest.approx((5 * 16 + 12) / 6)


def test_reference_is_capped_per_hand_and_saturated_deck_gets_no_bonus(public_state):
    public_state['global']['deck'] = [('Ghostly Armor', 1)] * 6
    assert hand_block_coverage(tuple(public_state['global']['deck']), 2, 14) == 14
    bonus, details = reward_adjustment(('Flame Barrier', 1), public_state)
    assert bonus == 0
    assert details['before'] == details['after'] == 14


def test_single_use_status_and_hand_consumption_are_not_free_recurring_block(public_state):
    for cid in ('Impervious', 'Power Through', 'True Grit'):
        assert hand_block_coverage(((cid, 1),) * 6, 4, 99) == 0
        assert reward_adjustment((cid, 1), public_state) == (0, None)
    # Their v1 acquisition values are preserved, not assigned zero overall.
    public_state['options']['cards'] = [('Impervious', 0), ('Power Through', 0)]
    legal = [A._RCARD, A._RCARD + 1, A._RSKIP]
    assert rank_demand_rewards(public_state, legal)[0] == [
        dict(row, v1_score=row['score'], demand=None) if A.segment_of(row['action'])[0] == 'REWARD_CARD' else row
        for row in rank_overworld(public_state, legal)]


def test_true_grit_fuel_counterexample_is_excluded_from_this_feature():
    # Five True Grits + four energy can only play three (the others are fuel).
    # Exclude this converter entirely rather than claim the energy-only 4*7
    # relaxation is executable. Its acquisition value still comes from v1.
    assert hand_block_coverage((('True Grit', 0),) * 5, 4, 99) == 0


@pytest.mark.parametrize('boss,reference', [('AUTOMATON', 14), ('CHAMP', 16), ('COLLECTOR', 18)])
def test_public_native_encounter_id_selects_reference(public_state, boss, reference):
    public_state['global']['boss'] = boss
    profile = demand_profile(public_state)
    assert profile['reference'] == reference
    assert profile['energy_budget'] == 2
    public_state['global']['ascension'] = 4
    assert demand_profile(public_state)['reference'] > reference


def test_permanent_energy_and_boss_conditional_energy_are_public(public_state):
    public_state['global']['relics'] = ['Slavers Collar', 'Cursed Key', 'Lantern', 'Oddly Smooth Stone']
    profile = demand_profile(public_state)
    assert profile['energy_budget'] == 4  # three base + two persistent, minus one
    assert profile['dexterity'] == 1


@pytest.mark.parametrize('engine', ['Corruption', 'Feel No Pain', 'Barricade', 'Second Wind', 'Metallicize'])
def test_owned_unmodeled_defense_does_not_produce_false_shortage(public_state, engine):
    public_state['global']['deck'].append((engine, 0))
    assert demand_profile(public_state) is None
    assert reward_adjustment(('Ghostly Armor', 1), public_state) == (0, None)


@pytest.mark.parametrize('relic', ['Snecko Eye', 'Runic Pyramid', 'Dead Branch', 'Calipers'])
def test_unmodeled_cost_hand_or_engine_relic_abstains(public_state, relic):
    public_state['global']['relics'] = [relic]
    assert demand_profile(public_state) is None


def test_rewards_change_when_defense_is_missing_and_respect_mask(public_state):
    legal = [A._RCARD, A._RCARD + 1, A._RSKIP]
    assert rank_overworld(public_state, legal)[0]['action'] == A._RCARD + 1
    assert rank_demand_rewards(public_state, legal)[0][0]['action'] == A._RCARD
    assert rank_demand_rewards(public_state, legal[1:])[0][0]['action'] == A._RCARD + 1


def test_candidate_is_independent_of_other_offers_deck_order_and_private_fields(public_state):
    before = deepcopy(public_state)
    expected = reward_adjustment(('Ghostly Armor', 0), public_state)
    public_state['options']['cards'] = [('Demon Form', 1), ('Offering', 1)]
    public_state['global']['deck'].reverse()
    public_state['future_draw_order'] = object()
    assert reward_adjustment(('Ghostly Armor', 0), public_state) == expected
    assert reward_adjustment(('Ghostly Armor', 1), public_state)[0] > expected[0]
    assert before['global']['deck'] == list(reversed(public_state['global']['deck']))


@pytest.mark.parametrize('phase', ['rest', 'card_select', 'shop', 'map'])
def test_other_decision_surfaces_are_not_scored(public_state, phase):
    public_state['phase'] = phase
    assert rank_demand_rewards(public_state, [A._RCARD]) is None


def request_for(state, legal, seed):
    mask = np.zeros(A.N_ACTIONS, dtype=bool)
    mask[legal] = True
    base = SimpleNamespace(obs_canonical=lambda: state, rmap=None, gc=SimpleNamespace(cur_map_node_y=0))
    env = SimpleNamespace(thick=base, action_masks=lambda: mask)
    env.unwrapped = env
    return DecisionRequest(env, {}, {'phase': state['phase']}, seed, 0)


def test_composition_preserves_batch_order_fallback_diagnostics_and_trace(public_state):
    class Fallback:
        name = 'sentinel'
        def decide(self, requests, *, check):
            assert [r.seed for r in requests] == [2, 4]
            return [Decision(A._EVENT + 1, self.name) for _ in requests]
    unknown = deepcopy(public_state)
    unknown['phase'], unknown['options'] = 'event', {'event': 'UNKNOWN_EVENT'}
    outside = deepcopy(public_state)
    outside['global']['act'] = 1
    legal = [A._RCARD, A._RCARD + 1, A._RSKIP]
    requests = [request_for(public_state, legal, 1), request_for(unknown, [A._EVENT + 1], 2),
                request_for(outside, legal, 3), request_for(unknown, [A._EVENT + 1], 4)]
    states_before = deepcopy([public_state, unknown, outside])
    traces = []
    strategy = EnemyDemandUpperStrategy(Fallback(), trace=traces.append)
    decisions = strategy.decide(requests)
    assert decisions[0].action == A._RCARD
    assert decisions[0].diagnostics['overworld_demand_changed_decisions'] == 1
    assert decisions[2].action == rank_overworld(outside, legal)[0]['action']
    assert decisions[1].diagnostics['overworld_fallback_decisions'] == 1
    assert decisions[3].diagnostics['overworld_fallback_decisions'] == 1
    assert sorted(t['seed'] for t in traces) == [1, 2, 3, 4]
    assert all(t['kind'] == 'overworld_decision' and 'action' in t for t in traces)
    assert [public_state, unknown, outside] == states_before
    assert strategy_manifest(strategy)['fallback']['name'] == 'sentinel'
    assert strategy.decide([]) == []
