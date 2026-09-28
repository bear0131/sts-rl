"""Hand-computable resource proxies and unchanged phase ownership."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from env.core import action_space as A
from rl.evaluation.baselines.overworld_energy import (
    EnergyAwareUpperStrategy, affordable_prior_curve, energy_profile, rank_energy_relics,
)
from rl.evaluation.baselines.overworld_rules import OverworldRuleStrategy, rank_overworld
from rl.evaluation.strategies import Decision, DecisionRequest, UPPER_STRATEGIES, strategy_manifest


def state(*, deck=None, relics=(), offers=('Black Blood', 'Sozu'), phase='boss_relic'):
    return dict(phase=phase, options={}) | {'global': dict(act=1, floor=17,
        hp=65, max_hp=80, gold=120, relics=list(relics), potions=[''] * 3,
        boss_relics=list(offers), deck=list(deck if deck is not None else
            [('Bash', 1), ('Uppercut', 1), ('Flame Barrier', 1), ('Shrug It Off', 1), ('Ghostly Armor', 0)]))}


def rows(s):
    return rank_energy_relics(s, [A._BOSS + i for i in range(len(s['global']['boss_relics']))])[0]


def detail(s, relic='Sozu'):
    return next(r['energy'] for r in rows(s) if r['reason'] == relic)


def test_uniform_probability_and_no_free_extra_card():
    cards = ((1, 9.0),) + ((0, 0.0),) * 5
    assert affordable_prior_curve(cards, 2) == (0, 7.5, 7.5)


def test_second_expensive_card_needs_two_extra_energy():
    cards = ((2, 10.0),) * 2 + ((0, 0.0),) * 4
    curve = affordable_prior_curve(cards, 4)
    assert curve[:4] == (0, 0, 10, 10)
    assert curve[4] == pytest.approx(50 / 3)  # 4/6 contain both, 2/6 one.


def test_zero_cost_cards_do_not_imply_energy_demand():
    assert affordable_prior_curve(((0, 3.0),) * 5, 4) == (15,) * 5
    assert affordable_prior_curve(((1, 3.0),) * 5, 4) == (0, 3, 6, 9, 12)


def test_pathological_diverse_deck_hits_bounded_abstention():
    assert affordable_prior_curve(tuple((1, float(i)) for i in range(1, 101)), 4) is None


def test_card_cost_and_order_are_public_and_immutable():
    s = state()
    before = deepcopy(s)
    expected = energy_profile(s)
    s['global']['deck'].reverse()
    s['hidden_next_draw'] = object()
    assert energy_profile(s) == expected
    assert before['global']['deck'] == list(reversed(s['global']['deck']))
    upgraded = state(deck=[('Dark Embrace', 1)])
    assert energy_profile(upgraded)['cards'][0][0] == 1
    assert energy_profile(state(deck=[('Dark Embrace', 0)]))['cards'][0][0] == 2


def test_capacity_correction_is_deck_dependent_and_can_change_a_real_choice():
    rich = state()
    assert rank_overworld(rich, [A._BOSS, A._BOSS + 1])[0]['action'] == A._BOSS
    assert rows(rich)[0]['action'] == A._BOSS + 1
    cheap = state(deck=[('Anger', 0)] * 5)
    assert detail(cheap)['correction'] == 0
    assert rows(cheap)[0]['action'] == A._BOSS
    assert detail(rich)['correction'] > detail(cheap)['correction']


def test_existing_energy_changes_the_marginal_budget_not_a_fixed_bonus():
    s = state(deck=[('Shrug It Off', 0)] * 5, relics=['Cursed Key', 'Fusion Hammer'])
    assert detail(s)['base_energy'] == 5
    assert detail(s)['correction'] == 0  # all five cards already fit.
    assert detail(state(deck=s['global']['deck']))['correction'] > 0


def test_collar_is_conditional_and_does_not_become_permanent_energy():
    s = state(offers=('Sozu', 'SlaversCollar'))
    assert detail(s, 'SlaversCollar')['correction'] == pytest.approx(detail(s)['correction'] / 2)
    held = state(relics=['SlaversCollar'])
    assert energy_profile(held)['energy'] == 3
    assert energy_profile(held)['collar'] == 1


def test_mark_of_pain_wounds_keep_draw_tax_even_with_medical_kit():
    s = state(offers=('Sozu', 'Mark of Pain'))
    pain = detail(s, 'Mark of Pain')
    assert pain['wound_filler'] == 2 and pain['correction'] < detail(s)['correction']
    s['global']['relics'] = ['Medical Kit']
    medical = detail(s, 'Mark of Pain')
    assert medical['medical_kit'] and medical['correction'] == pain['correction']
    assert affordable_prior_curve(((1, 8.0),) * 5 + ((0, 0.0),) * 2, 5)[5] == pytest.approx(200 / 7)
    owned = energy_profile(state(relics=['Mark of Pain']))
    assert owned['energy'] == 4 and len(owned['cards']) == 7


def test_draw_cards_keep_prior_but_do_not_create_unpaid_sixth_card():
    s = state(deck=[('Pommel Strike', 1), ('Shrug It Off', 1), ('Battle Trance', 0), ('Defend_R', 0), ('Strike_R', 0)])
    profile = energy_profile(s)
    assert profile['draw_cards'] == 3
    assert len(profile['cards']) == 5
    curve = affordable_prior_curve(profile['cards'], 5)
    assert curve[4] == curve[5]  # four positive-cost cards + free Trance.


@pytest.mark.parametrize('card', ['Offering', 'Bloodletting', 'Seeing Red', 'Corruption', 'Berserk', 'Whirlwind', 'UnknownCard'])
def test_unmodeled_temporary_energy_cost_or_unknown_card_abstains(card):
    assert energy_profile(state(deck=[(card, 0)])) is None


@pytest.mark.parametrize('relic', ['Snecko Eye', 'Runic Pyramid', 'Ice Cream', 'Velvet Choker'])
def test_unmodeled_hand_and_card_cap_abstains(relic):
    assert rank_energy_relics(state(relics=[relic]), [A._BOSS, A._BOSS + 1]) is None


def test_unmodeled_relic_penalties_remain_v1_and_legal_subset_is_respected():
    s = state(offers=('Sozu', 'Velvet Choker', 'Runic Dome'))
    ranked = rows(s)
    base = rank_overworld(s, [A._BOSS, A._BOSS + 1, A._BOSS + 2])
    for name in ('Velvet Choker', 'Runic Dome'):
        assert next(r for r in ranked if r['reason'] == name) == next(r for r in base if r['reason'] == name)
    assert rank_energy_relics(s, [A._BOSS + 1]) is None
    assert rank_energy_relics(s, [A._BOSS])[0][0]['action'] == A._BOSS


@pytest.mark.parametrize('phase', ['combat_reward', 'rest', 'card_select', 'map', 'shop'])
def test_no_changes_to_other_upper_decisions(phase):
    assert rank_energy_relics(state(phase=phase), [A._BOSS]) is None


def request(s, legal, seed):
    mask = np.zeros(A.N_ACTIONS, bool)
    mask[legal] = True
    env = SimpleNamespace(thick=SimpleNamespace(obs_canonical=lambda: s, rmap=None, gc=SimpleNamespace(cur_map_node_y=0)), action_masks=lambda: mask)
    env.unwrapped = env
    return DecisionRequest(env, {}, {'phase': s['phase']}, seed, 0)


def test_decision_batch_fallback_trace_and_factory_metadata():
    class Fallback:
        name = 'supplied-upper-sentinel'
        def decide(self, requests, *, check):
            assert [r.seed for r in requests] == [2]
            return [Decision(A._EVENT, self.name)]
    boss = state()
    unknown = state(phase='event')
    unknown['options'] = {'event': 'UnknownEvent'}
    traces = []
    policy = EnergyAwareUpperStrategy(Fallback(), trace=traces.append)
    before = deepcopy(boss)
    decisions = policy.decide([request(boss, [A._BOSS, A._BOSS + 1], 1), request(unknown, [A._EVENT], 2)])
    assert decisions[0].action == A._BOSS + 1
    assert decisions[0].diagnostics['overworld_energy_changed_decisions'] == 1
    assert decisions[1].diagnostics['overworld_fallback_decisions'] == 1
    assert [t['seed'] for t in traces] == [1, 2]
    assert all(t['kind'] == 'overworld_decision' and 'action' in t for t in traces)
    assert boss == before
    manifest = strategy_manifest(policy)
    assert manifest['scope'] == 'boss-relic-only' and manifest['base_version'] == 'ironclad-overworld-v3.1-potion-contracts'
    assert manifest['fallback']['name'] == Fallback.name
    model = object()
    made = UPPER_STRATEGIES['energy'](model, SimpleNamespace(stochastic=False))
    assert made.fallback.model is model
    assert made._base.name == 'overworld-rules'
    assert policy.decide([]) == []


def test_reward_delegation_is_exact_v1_and_boss_abstention_is_counted():
    reward = state(phase='combat_reward')
    reward['options'] = {'cards': [('Ghostly Armor', 0), ('Battle Trance', 0)]}
    req = request(reward, [A._RCARD, A._RCARD + 1, A._RSKIP], 1)
    assert EnergyAwareUpperStrategy().decide([req]) == OverworldRuleStrategy().decide([req])
    boss = state(relics=['Snecko Eye'])
    req = request(boss, [A._BOSS, A._BOSS + 1], 1)
    candidate = EnergyAwareUpperStrategy().decide([req])[0]
    baseline = OverworldRuleStrategy().decide([req])[0]
    assert candidate.action == baseline.action and candidate.source == baseline.source
    assert candidate.diagnostics['overworld_energy_abstentions'] == 1
