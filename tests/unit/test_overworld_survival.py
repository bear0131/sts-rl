"""Public-state behavior contracts for the opt-in A20 upper challenger."""
from copy import deepcopy
from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import numpy as np
import pytest

from env.core import action_space as A
from rl.evaluation.baselines.overworld_survival import (
    GROWTH_PROFILE, READINESS_PROFILE, GrowthUpperStrategy, ReadinessUpperStrategy,
    SurvivalUpperStrategy, coverage, elite_ready, rank_survival,
    survival_card_value, survival_route_scores,
)
from rl.evaluation.strategies import Decision, DecisionRequest


@pytest.fixture
def make_state():
    def make(phase='combat_reward', *, cards=(), deck=(), act=1, floor=3, hp=80, relics=()):
        starter = [('Strike_R', 0)] * 5 + [('Defend_R', 0)] * 4 + [('Bash', 0), ('AscendersBane', 0)]
        return dict(phase=phase, options=dict(cards=list(cards))) | {
            'global': dict(hp=hp, max_hp=80, gold=100, act=act, floor=floor,
                ascension=20, deck=starter + list(deck), relics=list(relics),
                potions=['EMPTY_POTION_SLOT'] * 2)}
    return make


def choose(state, *actions, **kwargs):
    return rank_survival(state, actions, **kwargs)[0]['action']


def test_early_damage_beats_unprepared_engine_and_ascenders_bane_breaks_clash(make_state):
    state = make_state(cards=[('Demon Form', 0), ('Carnage', 0), ('Clash', 0)])
    assert choose(state, A._RCARD, A._RCARD + 1, A._RCARD + 2, A._RSKIP) == A._RCARD + 1
    assert survival_card_value(('Clash', 1), state) < 0


def test_act2_missing_area_damage_beats_another_single_target_attack(make_state):
    state = make_state(act=2, floor=18,
        deck=[('Carnage', 1), ('Hemokinesis', 1), ('Anger', 0), ('Pommel Strike', 0)],
        cards=[('Cleave', 0), ('Carnage', 0)])
    assert choose(state, A._RCARD, A._RCARD + 1, A._RSKIP) == A._RCARD
    without = survival_card_value(('Cleave', 0), state)
    state['global']['deck'].append(('Immolate', 0))
    assert survival_card_value(('Cleave', 0), state) < without - 3


def test_act2_repair_block_before_redundant_attacks(make_state):
    state = make_state(act=2, floor=21,
        deck=[('Immolate', 0), ('Carnage', 0), ('Hemokinesis', 0)],
        cards=[('Ghostly Armor', 0), ('Twin Strike', 1)])
    assert choose(state, A._RCARD, A._RCARD + 1, A._RSKIP) == A._RCARD


def test_defensive_power_support_and_strength_are_required_for_payoffs(make_state):
    state = make_state(act=2)
    before = survival_card_value(('Limit Break', 0), state)
    state['global']['deck'].append(('Inflame', 0))
    assert survival_card_value(('Limit Break', 0), state) > max(0, before + 8)
    assert survival_card_value(('Entrench', 0), state) < 0
    state['global']['deck'].append(('Barricade', 0))
    assert survival_card_value(('Entrench', 1), state) > 0


def test_power_through_second_wind_is_more_than_flat_card_ranking(make_state):
    state = make_state(act=2)
    before = survival_card_value(('Second Wind', 0), state)
    state['global']['deck'].append(('Power Through', 0))
    assert survival_card_value(('Second Wind', 0), state) > before


def test_covered_large_deck_skips_redundant_plain_damage(make_state):
    state = make_state(act=3, floor=44, cards=[('Twin Strike', 0), ('Heavy Blade', 0)],
        deck=[('Immolate', 1), ('Carnage', 1), ('Hemokinesis', 1), ('Pommel Strike', 1),
              ('Shrug It Off', 1), ('Flame Barrier', 1), ('True Grit', 1), ('Impervious', 0)])
    assert choose(state, A._RCARD, A._RCARD + 1, A._RSKIP) == A._RSKIP


def test_injured_act2_rests_but_healthy_player_upgrades(make_state):
    state = make_state('rest', act=2, hp=47, deck=[('Carnage', 0)])
    assert choose(state, A._CAMP, A._CAMP + 1, current_y=8) == A._CAMP
    state['global']['hp'] = 79
    assert choose(state, A._CAMP, A._CAMP + 1, current_y=8) == A._CAMP + 1


def test_card_select_uses_displayed_candidates_and_remove_curse_is_retained(make_state):
    state = make_state('card_select', cards=[('Strike_R', 0), ('Carnage', 0), ('Normality', 0)])
    state['options']['select'] = {'context': 'UPGRADE'}
    assert choose(state, A._PICK, A._PICK + 1) == A._PICK + 1
    state['options']['select']['context'] = 'REMOVE'
    assert choose(state, A._PICK, A._PICK + 2) == A._PICK + 2


def test_shop_buys_useful_potion_and_guards_both_known_native_waste_cases(make_state):
    state = make_state('shop', act=2)
    state['options']['shop_potions'] = [('Block Potion', 50)]
    assert choose(state, A._SHOP_POTION, A._PROCEED) == A._SHOP_POTION
    state['global']['relics'] = ['Sozu']
    assert choose(state, A._SHOP_POTION, A._PROCEED) == A._PROCEED
    state['global']['relics'] = []
    state['global']['potions'] = ['Fire Potion', 'Block Potion']
    assert choose(state, A._SHOP_POTION, A._PROCEED) == A._PROCEED
    state['global']['potions'] = ['INVALID', 'INVALID']
    assert choose(state, A._SHOP_POTION, A._PROCEED) == A._PROCEED


def test_shop_does_not_spend_starting_gold_removing_damage_before_obtaining_any(make_state):
    state = make_state('shop', cards=[])
    state['options'].update(shop_cards=[('Carnage', 0, 75)], shop_remove_cost=75)
    assert choose(state, A._SHOP_CARD, A._SHOP_REMOVE, A._PROCEED) == A._SHOP_CARD


def test_removable_curse_still_beats_noop_early_shop(make_state):
    state = make_state('shop', deck=[('Normality', 0)])
    state['options']['shop_remove_cost'] = 75
    assert choose(state, A._SHOP_REMOVE, A._PROCEED) == A._SHOP_REMOVE


def test_cumulative_elite_risk_depends_on_rest_order_not_only_room_counts(make_state):
    class Map:
        rooms = {0: ['ELITE', 'ELITE', 'REST'], 1: ['ELITE', 'REST', 'ELITE']}
        def room_type(self, x, y):
            return self.rooms[x][y]
        def reachable(self, x, y):
            return [x] if y < 2 else []
    state = make_state('map', act=2, hp=45,
        deck=[('Immolate', 0), ('Shrug It Off', 0), ('Power Through', 0), ('Flame Barrier', 0)])
    scores = survival_route_scores(state, [(0, 100), (1, 101)], Map(), -1)
    assert scores[101] > scores[100]


def test_ready_healthy_player_can_take_elite_but_low_health_seeks_recovery(make_state):
    class Map:
        def room_type(self, x, y):
            return 'ELITE' if x == 0 else 'REST'
        def reachable(self, x, y):
            return []
    state = make_state('map', deck=[('Carnage', 1), ('Hemokinesis', 1), ('Immolate', 1)])
    scores = survival_route_scores(state, [(0, 100), (1, 101)], Map(), 8)
    assert scores[100] > scores[101]
    state['global']['hp'] = 25
    scores = survival_route_scores(state, [(0, 100), (1, 101)], Map(), 8)
    assert scores[101] > scores[100]


def test_unavailable_heal_and_smith_do_not_attract_to_rest(make_state):
    class Map:
        def room_type(self, x, y):
            return 'REST' if x == 0 else 'EVENT'
        def reachable(self, x, y):
            return []
    state = make_state('map', hp=25, relics=['Coffee Dripper', 'Fusion Hammer'])
    scores = survival_route_scores(state, [(0, 100), (1, 101)], Map(), 8)
    assert scores[101] > scores[100]


def test_strategy_is_deterministic_nonmutating_and_reports_version(make_state):
    state = make_state(cards=[('Carnage', 0), ('Demon Form', 0)])
    before = deepcopy(state)
    mask = np.zeros(A.N_ACTIONS, dtype=bool)
    mask[[A._RCARD, A._RCARD + 1, A._RSKIP]] = True
    base = SimpleNamespace(obs_canonical=lambda: state, rmap=None, gc=SimpleNamespace(cur_map_node_y=1))
    env = SimpleNamespace(thick=base, action_masks=lambda: mask)
    env.unwrapped = env
    request = DecisionRequest(env, {}, {'phase': state['phase']}, 42, 0)
    traces = []
    strategy = SurvivalUpperStrategy(trace=traces.append)
    assert strategy.decide([request]) == strategy.decide([request])
    assert state == before
    assert traces[0]['version'] == strategy.version
    assert traces[0]['coverage']['frontload'] == 0


def test_unknown_event_falls_back_in_batch_and_combat_is_rejected(make_state):
    state = make_state('event')
    state['options']['event'] = 'UNKNOWN_EVENT'
    mask = np.zeros(A.N_ACTIONS, dtype=bool)
    mask[A._EVENT] = True
    base = SimpleNamespace(obs_canonical=lambda: state, rmap=None, gc=SimpleNamespace(cur_map_node_y=1))
    env = SimpleNamespace(thick=base, action_masks=lambda: mask)
    env.unwrapped = env
    requests = [DecisionRequest(env, {}, {}, 42, i) for i in range(2)]
    class Fallback:
        def decide(self, requests, *, check):
            return [Decision(A._EVENT, 'sentinel') for _ in requests]
    decisions = SurvivalUpperStrategy(Fallback()).decide(requests)
    assert all(d.action == A._EVENT and d.diagnostics['overworld_fallback_decisions'] == 1 for d in decisions)
    state['phase'] = 'combat'
    with pytest.raises(ValueError, match='combat strategy'):
        rank_survival(state, [A.END_TURN])


def test_boss_relic_avoids_sozu_when_healing_upgrade_available(make_state):
    state = make_state('boss_relic')
    state['global']['boss_relics'] = ['Sozu', 'Black Blood']
    assert choose(state, A._BOSS, A._BOSS + 1) == A._BOSS + 1
    state['global']['relics'] = ['Cursed Key']
    assert coverage(state).energy == 4


def test_growth_spends_prepared_hp_on_elite_where_survival_prefers_rest(make_state):
    class Map:
        def room_type(self, x, y):
            return 'ELITE' if x == 0 else 'REST'
        def reachable(self, x, y):
            return []
    state = make_state('map', act=2, hp=60, deck=[('Immolate', 0), ('Carnage', 0),
        ('Pommel Strike', 0), ('Shrug It Off', 0), ('Power Through', 0),
        ('Flame Barrier', 0), ('Shockwave', 0)])
    assert elite_ready(state)
    survival = survival_route_scores(state, [(0, 100), (1, 101)], Map(), 8)
    growth = survival_route_scores(state, [(0, 100), (1, 101)], Map(), 8, profile=GROWTH_PROFILE)
    assert survival[101] > survival[100]
    assert growth[100] > growth[101]
    state['global']['hp'] = 20
    growth = survival_route_scores(state, [(0, 100), (1, 101)], Map(), 8, profile=GROWTH_PROFILE)
    assert growth[101] > growth[100]


def test_growth_never_unlocks_elite_reward_for_unprepared_starter(make_state):
    class Map:
        def room_type(self, x, y):
            return 'ELITE'
        def reachable(self, x, y):
            return []
    state = make_state('map', hp=80)
    assert not elite_ready(state)
    survival = survival_route_scores(state, [(0, 100)], Map(), 8)
    growth = survival_route_scores(state, [(0, 100)], Map(), 8, profile=GROWTH_PROFILE)
    assert growth == survival


def test_growth_potion_readiness_is_consumable_in_projected_elite_chain(make_state):
    state = make_state('map', deck=[('Carnage', 0), ('Pommel Strike', 0)])
    assert not elite_ready(state)
    state['global']['potions'][0] = 'Fire Potion'
    assert elite_ready(state)
    assert not elite_ready(state, potion_reserves=0)
    class Map:
        def room_type(self, x, y):
            return 'ELITE' if y < 2 else 'REST'
        def reachable(self, x, y):
            return [x] if y < 2 else []
    one = survival_route_scores(state, [(0, 100)], Map(), -1, profile=GROWTH_PROFILE)[100]
    state['global']['potions'][1] = 'Block Potion'
    two = survival_route_scores(state, [(0, 100)], Map(), -1, profile=GROWTH_PROFILE)[100]
    assert two > one


def test_growth_upgrades_at_moderate_hp_but_heals_before_boss_or_at_death_risk(make_state):
    state = make_state('rest', act=2, hp=47, deck=[('Carnage', 0)])
    assert choose(state, A._CAMP, A._CAMP + 1, current_y=8) == A._CAMP
    assert choose(state, A._CAMP, A._CAMP + 1, current_y=8, profile=GROWTH_PROFILE) == A._CAMP + 1
    assert choose(state, A._CAMP, A._CAMP + 1, current_y=14, profile=GROWTH_PROFILE) == A._CAMP
    state['global']['hp'] = 20
    assert choose(state, A._CAMP, A._CAMP + 1, current_y=8, profile=GROWTH_PROFILE) == A._CAMP


def test_growth_profile_is_immutable_and_card_acquisition_is_controlled(make_state):
    with pytest.raises(FrozenInstanceError):
        GROWTH_PROFILE.elite_growth_bonus = 999
    state = make_state(cards=[('Carnage', 0), ('Demon Form', 0)])
    actions = [A._RCARD, A._RCARD + 1, A._RSKIP]
    assert rank_survival(state, actions) == rank_survival(state, actions, profile=GROWTH_PROFILE)
    assert GrowthUpperStrategy().manifest()['profile']['name'] == 'growth'
    assert SurvivalUpperStrategy().manifest()['profile']['name'] == 'survival'


def test_readiness_does_not_commit_unprepared_deck_to_back_to_back_elites(make_state):
    class Map:
        rooms = {0: ['ELITE', 'MONSTER', 'ELITE'], 1: ['REST', 'ELITE', 'MONSTER']}
        def room_type(self, x, y):
            return self.rooms[x][y]
        def reachable(self, x, y):
            return [x] if y < 2 else []
    state = make_state('map', hp=56, relics=['Burning Blood'],
        deck=[('Carnage', 0), ('Pommel Strike', 0), ('True Grit', 0)])
    scores = survival_route_scores(state, [(0, 100), (1, 101)], Map(), -1,
                                   profile=READINESS_PROFILE)
    assert not elite_ready(state)
    assert scores[101] > scores[100]


def test_readiness_preserves_full_elite_reward_for_prepared_healthy_deck(make_state):
    class Map:
        def room_type(self, x, y):
            return 'ELITE' if x == 0 else 'EVENT'
        def reachable(self, x, y):
            return []
    state = make_state('map', deck=[('Carnage', 1), ('Pommel Strike', 1), ('Immolate', 1)])
    before = survival_route_scores(state, [(0, 100), (1, 101)], Map(), 5)
    after = survival_route_scores(state, [(0, 100), (1, 101)], Map(), 5,
                                  profile=READINESS_PROFILE)
    assert after == before
    assert after[100] > after[101]
    state['global']['hp'] = 32
    after = survival_route_scores(state, [(0, 100), (1, 101)], Map(), 5,
                                  profile=READINESS_PROFILE)
    assert after[101] > after[100]


def test_readiness_profile_changes_only_routes(make_state):
    for phase, actions in [('combat_reward', [A._RCARD, A._RCARD + 1, A._RSKIP]),
                           ('rest', [A._CAMP, A._CAMP + 1])]:
        state = make_state(phase, hp=47, act=2, cards=[('Carnage', 0), ('Demon Form', 0)],
                           deck=[('Carnage', 0)])
        assert rank_survival(state, actions) == rank_survival(state, actions, profile=READINESS_PROFILE)
    assert ReadinessUpperStrategy().manifest()['profile']['elite_policy'] == 'readiness'
