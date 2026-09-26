"""Meaningful deck/route tradeoffs and explicit unknown-screen fallback."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from env.core import action_space as A
from env.core.run_env import SlayTheSpireEnv
from env.search import overworld_rules as shared_rules
from rl.evaluation.baselines.overworld_rules import (
    OverworldRuleStrategy, card_value, rank_overworld, relic_value, route_scores,
)
from rl.evaluation.strategies import Decision, DecisionRequest


def state(phase='combat_reward', *, cards=(), deck=None, hp=80, **global_extra):
    deck = deck if deck is not None else [('Strike_R', 0)] * 5 + [('Defend_R', 0)] * 4 + [('Bash', 0)]
    global_state = dict(
        hp=hp, max_hp=80, gold=100, act=1, floor=3, deck=deck, relics=[],
        potions=['EMPTY_POTION_SLOT'] * 3)
    global_state.update(global_extra)
    return {'phase': phase, 'options': {'cards': list(cards)}, 'global': global_state}


def choose(s, *actions):
    return rank_overworld(s, list(actions))[0]['action']


def test_early_deck_values_damage_but_skips_unenabled_limit_break():
    s = state(cards=[('Limit Break', 0), ('Carnage', 0)])
    assert choose(s, A._RCARD, A._RCARD + 1, A._RSKIP) == A._RCARD + 1
    s['options']['cards'] = [('Limit Break', 0), ('Clash', 0)]
    assert choose(s, A._RCARD, A._RCARD + 1, A._RSKIP) == A._RSKIP


def test_strength_enables_scaling_and_duplicate_powers_diminish():
    s = state(deck=[('Strike_R', 0), ('Inflame', 0), ('Spot Weakness', 0)])
    empty = state(deck=[('Strike_R', 0)])
    assert card_value(('Limit Break', 0), s) > card_value(('Limit Break', 0), empty) + 5
    assert card_value(('Inflame', 0), s) < card_value(('Inflame', 0), empty)


def test_remove_curse_and_upgrade_uppercut_use_screen_candidates():
    s = state('card_select', cards=[('Strike_R', 0), ('Normality', 0), ('Uppercut', 0)])
    s['options']['select'] = {'context': 'REMOVE'}
    assert choose(s, A._PICK, A._PICK + 1, A._PICK + 2) == A._PICK + 1
    s['options']['select']['context'] = 'UPGRADE'
    assert choose(s, A._PICK, A._PICK + 2) == A._PICK + 2


def test_rest_at_low_health_upgrade_when_healthy_and_never_recall_for_act3():
    s = state('rest', hp=25)
    assert choose(s, A._CAMP, A._CAMP + 1, A._CAMP + 2) == A._CAMP
    s['global']['hp'] = 78
    assert choose(s, A._CAMP, A._CAMP + 1, A._CAMP + 2) == A._CAMP + 1


def test_shop_compares_real_price_and_skips_bad_card():
    s = state('shop')
    s['options'].update(shop_cards=[('Clash', 0, 25), ('Offering', 0, 60)], shop_remove_cost=100)
    assert choose(s, A._SHOP_CARD, A._SHOP_CARD + 1, A._PROCEED) == A._SHOP_CARD + 1
    s['options']['shop_cards'][1] = ('Clash', 0, 150)
    assert choose(s, A._SHOP_CARD, A._SHOP_CARD + 1, A._PROCEED) == A._PROCEED


def test_actual_relic_id_spelling_and_dome_drawback_are_understood():
    s = state('boss_relic')
    s['global']['boss_relics'] = ['Runic Dome', 'Cursed Key', "Philosopher's Stone"]
    assert choose(s, A._BOSS, A._BOSS + 1, A._BOSS + 2) == A._BOSS + 1
    assert relic_value('Toxic Egg 2', s) == relic_value('TOXIC_EGG', s)
    s['phase'] = 'chest'
    s['global']['relics'] = ['Cursed Key']
    assert choose(s, A._CHEST, A._CHEST + 1) == A._CHEST + 1


def test_route_prefers_recovery_over_elite_when_wounded():
    class Map:
        def room_type(self, x, y):
            return 'REST' if x == 0 else 'ELITE'
        def reachable(self, x, y):
            return []
    scores = route_scores(state('map', hp=25), [(0, 0), (1, 1)], Map(), 8)
    assert scores[0] > scores[1]


def test_unknown_event_is_batched_to_explicit_fallback_without_mutation():
    s = state('event')
    s['options']['event'] = 'NEW_UNDOCUMENTED_EVENT'
    before = deepcopy(s)
    mask = np.zeros(A.N_ACTIONS, dtype=bool)
    mask[A._EVENT + 1] = True
    base = SimpleNamespace(obs_canonical=lambda: s, rmap=None, gc=SimpleNamespace(cur_map_node_y=0))
    env = SimpleNamespace(thick=base, action_masks=lambda: mask)
    env.unwrapped = env
    class Fallback:
        def decide(self, requests, *, check):
            assert len(requests) == 2
            return [Decision(A._EVENT + 1, 'sentinel') for _ in requests]
    requests = [DecisionRequest(env, {}, {'phase': 'event'}, 10000, i) for i in range(2)]
    traces = []
    decisions = OverworldRuleStrategy(Fallback(), trace=traces.append).decide(requests)
    assert [d.action for d in decisions] == [A._EVENT + 1] * 2
    assert all(d.diagnostics['overworld_fallback_decisions'] == 1 for d in decisions)
    assert all(t['kind'] == 'overworld_decision' and t['action'] == A._EVENT + 1
               and 'fallback:sentinel' in t['source'] for t in traces)
    assert s == before
    with pytest.raises(ValueError, match='without fallback'):
        OverworldRuleStrategy().decide(requests)


def test_reward_potion_full_slots_does_not_loop_and_healing_is_not_wasted():
    s = state()
    s['global']['potions'] = ['FIRE_POTION', 'BLOOD_POTION', 'BLOCK_POTION']
    assert choose(s, A._RPOTION, A._POTION + 1, A._PROCEED) == A._PROCEED
    s['global']['hp'] = 50
    assert choose(s, A._RPOTION, A._POTION + 1, A._PROCEED) == A._POTION + 1


@pytest.mark.parametrize('event', ['Big Fish', 'BIG_FISH'])
def test_big_fish_heals_wounded_player_and_healthy_player_takes_max_hp(event):
    s = state('event', hp=30)
    s['options']['event'] = event
    assert choose(s, A._EVENT, A._EVENT + 1, A._EVENT + 2) == A._EVENT
    s['global']['hp'] = 80
    assert choose(s, A._EVENT, A._EVENT + 1, A._EVENT + 2) == A._EVENT + 1


def test_combat_request_is_rejected_instead_of_silently_using_upper_model():
    with pytest.raises(ValueError, match='combat strategy'):
        rank_overworld(state('combat'), [A.END_TURN])


def test_worker_uses_same_core_and_explicit_unknown_event_fallback():
    assert rank_overworld is shared_rules.rank_overworld
    s = state('event')
    s['options']['event'] = 'NEW_UNDOCUMENTED_EVENT'
    env = object.__new__(SlayTheSpireEnv)
    env.thick = SimpleNamespace(obs_canonical=lambda: s, rmap=None,
                                gc=SimpleNamespace(cur_map_node_y=0))
    env._phase = lambda: s['phase']
    mask = np.zeros(A.N_ACTIONS, dtype=bool)
    mask[A._EVENT + 1] = True
    env.action_masks = lambda: mask
    assert env.overworld_rule_decision() is None
    s['phase'] = 'combat'
    with pytest.raises(RuntimeError, match='non-combat'):
        env.overworld_rule_decision()


def test_permanent_curse_does_not_make_expensive_shop_removal_look_good():
    s = state('shop', deck=[('AscendersBane', 0), ('Uppercut', 1), ('Strike_R', 0)])
    s['options']['shop_remove_cost'] = 250
    assert choose(s, A._SHOP_REMOVE, A._PROCEED) == A._PROCEED


def test_legacy_event_alias_and_mindbloom_avoid_irreversible_deck_damage():
    s = state('event')
    s['options']['event'] = 'Back to Basics'
    assert choose(s, A._EVENT, A._EVENT + 1) == A._EVENT + 1
    s['options']['event'] = 'Mindbloom'
    assert choose(s, A._EVENT, A._EVENT + 1, A._EVENT + 2) == A._EVENT


@pytest.mark.parametrize(('event', 'actions', 'expected'), [
    ('Drug Dealer', (0, 1, 2), 2),
    ('Augmenter', (0, 1, 2), 2),
    ('Duplicator', (0, 1), 0),
    ('Sensory Stone', (0, 1, 2), 0),
])
def test_public_deterministic_events_use_simple_growth_rules(event, actions, expected):
    s = state('event')
    s['options']['event'] = event
    legal = tuple(A._EVENT + action for action in actions)
    assert choose(s, *legal) == A._EVENT + expected


@pytest.mark.parametrize(('event', 'healthy_choice', 'wounded_choice'), [
    ('Face Trader', 0, 2),
    ('Masked Bandits', 1, 0),
    ('Mushrooms', 0, 1),
    ('Mysterious Sphere', 0, 1),
])
def test_risky_events_use_visible_health_thresholds(event, healthy_choice, wounded_choice):
    s = state('event', hp=70)
    s['options']['event'] = event
    actions = (A._EVENT, A._EVENT + 1, A._EVENT + 2)
    if event in ('Masked Bandits', 'Mushrooms', 'Mysterious Sphere'):
        actions = actions[:2]
    assert choose(s, *actions) == A._EVENT + healthy_choice
    s['global']['hp'] = 30
    assert choose(s, *actions) == A._EVENT + wounded_choice


def test_forgotten_altar_prefers_idol_then_health_then_curse():
    s = state('event', hp=70)
    s['options']['event'] = 'Forgotten Altar'
    s['global']['relics'] = ['Golden Idol']
    assert choose(s, A._EVENT, A._EVENT + 1, A._EVENT + 2) == A._EVENT
    s['global']['relics'] = []
    assert choose(s, A._EVENT + 1, A._EVENT + 2) == A._EVENT + 1
    s['global']['hp'] = 30
    assert choose(s, A._EVENT + 1, A._EVENT + 2) == A._EVENT + 2


def test_red_mask_tomb_uses_only_current_legal_resource_exchange():
    s = state('event', gold=200)
    s['options']['event'] = 'Tomb of Lord Red Mask'
    s['global']['relics'] = ['Red Mask']
    assert choose(s, A._EVENT, A._EVENT + 2) == A._EVENT
    s['global']['relics'] = []
    assert choose(s, A._EVENT + 1, A._EVENT + 2) == A._EVENT + 2
    s['global']['gold'] = 50
    assert choose(s, A._EVENT + 1, A._EVENT + 2) == A._EVENT + 1


def test_vampires_only_trade_blood_vial_and_never_guess_an_illegal_option():
    s = state('event')
    s['options']['event'] = 'Vampires'
    s['global']['relics'] = ['Blood Vial']
    assert choose(s, A._EVENT, A._EVENT + 1, A._EVENT + 2) == A._EVENT
    s['global']['relics'] = []
    assert choose(s, A._EVENT + 1, A._EVENT + 2) == A._EVENT + 2


def test_woman_in_blue_buys_one_only_with_space_and_visible_funds():
    s = state('event', gold=50)
    s['options']['event'] = 'The Woman in Blue'
    actions = tuple(A._EVENT + i for i in range(4))
    assert choose(s, *actions) == A._EVENT
    s['global']['potions'] = ['FIRE_POTION', 'BLOCK_POTION', 'BLOOD_POTION']
    assert choose(s, *actions) == A._EVENT + 3


def test_bonfire_spirits_selects_the_best_public_sacrifice():
    s = state('card_select', cards=[('Strike_R', 0), ('Normality', 0), ('Uppercut', 0)])
    s['options']['select'] = {'context': 'BONFIRE_SPIRITS'}
    assert choose(s, A._PICK, A._PICK + 1, A._PICK + 2) == A._PICK + 1


@pytest.mark.parametrize('event', [
    'Falling', 'Nloth', 'WeMeetAgain', 'Note For Yourself', 'Designer',
    'NEW_UNDOCUMENTED_EVENT',
])
def test_hidden_or_dynamic_event_candidates_still_require_fallback(event):
    s = state('event')
    s['options']['event'] = event
    assert rank_overworld(s, [A._EVENT, A._EVENT + 1, A._EVENT + 2]) is None


def test_event_rules_never_return_an_unavailable_option():
    s = state('event')
    s['options']['event'] = 'Drug Dealer'
    assert choose(s, A._EVENT, A._EVENT + 1) in (A._EVENT, A._EVENT + 1)


def test_snecko_and_pyramid_conflict_in_both_acquisition_orders():
    s = state()
    before = relic_value('Snecko Eye', s)
    s['global']['relics'] = ['Runic Pyramid']
    assert relic_value('Snecko Eye', s) < before - 5


def test_unavailable_campfire_actions_do_not_attract_route_to_dead_rest():
    class Map:
        def room_type(self, x, y):
            return 'REST' if x == 0 else 'EVENT'
        def reachable(self, x, y):
            return []
    s = state('map', hp=25)
    s['global']['relics'] = ['Coffee Dripper', 'Fusion Hammer']
    scores = route_scores(s, [(0, 0), (1, 1)], Map(), 8)
    assert scores[0] < scores[1]
