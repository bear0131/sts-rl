"""Behavioral prerequisites and composition of the optional engine upper."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from env.core import action_space as A
from rl.evaluation.baselines.overworld_engine import (
    EngineUpperStrategy, boss_card_adjustment, engine_card_value, engine_profile, engine_relic_value,
    engine_remove_value, engine_skip_threshold, engine_upgrade_value,
    rank_engine_overworld,
)
from rl.evaluation.baselines.overworld_rules import rank_overworld
from rl.evaluation.strategies import Decision, DecisionRequest, strategy_manifest


@pytest.fixture
def state():
    return {'phase': 'combat_reward', 'options': {'cards': []}, 'global': dict(
        act=1, ascension=20, floor=3, hp=70, max_hp=80, gold=100,
        deck=[('Strike_R', 0)] * 5 + [('Defend_R', 0)] * 4 + [('Bash', 0), ('AscendersBane', 0)],
        relics=[], potions=['EMPTY_POTION_SLOT'] * 2)}


def choose(state, *cards):
    state['options']['cards'] = list(cards)
    legal = [A._RCARD + i for i in range(len(cards))] + [A._RSKIP]
    return rank_engine_overworld(state, legal)[0][0]['action']


def test_early_frontload_precedes_unsupported_engine_payoffs(state):
    assert choose(state, ('Limit Break', 0), ('Carnage', 0), ('Entrench', 0)) == A._RCARD + 1
    assert choose(state, ('Dark Embrace', 0), ('Clash', 0), ('Barricade', 0)) == A._RSKIP
    # The one Ascender's Bane exhaust is not a repeatable drawing engine.
    assert engine_profile(state).exhaust_fuel < 1


def test_strength_turns_limit_break_and_reaper_into_real_payoffs(state):
    empty_limit = engine_card_value(('Limit Break', 0), state)
    empty_reaper = engine_card_value(('Reaper', 0), state)
    state['global']['deck'] += [('Inflame', 0), ('Spot Weakness', 0)]
    assert engine_card_value(('Limit Break', 0), state) > empty_limit + 6
    assert engine_card_value(('Reaper', 0), state) > empty_reaper + 4
    state['global'].update(act=2, floor=21)
    assert choose(state, ('Limit Break', 0), ('Cleave', 0)) == A._RCARD


def test_exhaust_draw_requires_fuel_and_corruption_skills(state):
    bare = engine_card_value(('Dark Embrace', 0), state)
    corruption_bare = engine_card_value(('Corruption', 0), state)
    state['global']['deck'] += [('Disarm', 0), ('Offering', 0), ('Second Wind', 0),
        ('Shrug It Off', 0), ('Burning Pact', 0), ('Feel No Pain', 0)]
    assert engine_card_value(('Dark Embrace', 0), state) > bare + 4
    assert engine_card_value(('Corruption', 0), state) > corruption_bare + 4
    assert 'exhaust' in engine_profile(state).commitments
    # In Act 1 the attack-starved deck still needs immediate damage. Once its
    # frontload is present, fill the engine's missing draw piece.
    assert choose(state, ('Dark Embrace', 0), ('Carnage', 0)) == A._RCARD + 1
    state['global']['deck'] += [('Carnage', 0), ('Pommel Strike', 1)]
    assert choose(state, ('Dark Embrace', 0), ('Carnage', 0)) == A._RCARD


def test_dark_embrace_does_not_enable_an_exhume_without_a_target(state):
    state['global']['deck'] += [('Dark Embrace', 1)]
    assert choose(state, ('Exhume', 0)) == A._RSKIP
    state['global']['deck'] += [('Impervious', 0), ('Offering', 0)]
    assert choose(state, ('Exhume', 0)) == A._RCARD


def test_power_through_second_wind_pair_is_distinguished_from_random_defense(state):
    before = engine_card_value(('Second Wind', 0), state)
    state['global']['deck'] += [('Power Through', 0)]
    assert engine_card_value(('Second Wind', 0), state) > before + 1.5
    assert choose(state, ('Second Wind', 0), ('Ghostly Armor', 0)) == A._RCARD


def test_status_payoff_needs_status_generators_not_just_curses(state):
    assert choose(state, ('Fire Breathing', 0), ('Evolve', 0)) == A._RSKIP
    state['global']['deck'] += [('Power Through', 0), ('Immolate', 0)]
    assert choose(state, ('Evolve', 0), ('Clothesline', 0)) == A._RCARD
    before = engine_card_value(('Fire Breathing', 0), state)
    state['global']['deck'] += [('Evolve', 0)]
    assert engine_card_value(('Fire Breathing', 0), state) > before


def test_engine_discipline_skips_redundant_attack_and_duplicate_setup(state):
    state['global'].update(act=3, floor=42)
    state['global']['deck'] += [('Carnage', 0), ('Immolate', 0), ('Pommel Strike', 1),
        ('Corruption', 1), ('Dark Embrace', 1), ('Feel No Pain', 0), ('Shrug It Off', 0)] * 2
    assert choose(state, ('Cleave', 0), ('Demon Form', 0), ('Corruption', 0)) == A._RSKIP
    assert engine_skip_threshold(engine_profile(state)) > 8


def test_empty_deck_does_not_force_strength_or_exhaust_commitment(state):
    state['global']['deck'] = [('Strike_R', 0), ('Defend_R', 0)]
    assert not engine_profile(state).commitments
    assert choose(state, ('Limit Break', 1), ('Entrench', 1)) == A._RSKIP


def test_upgrades_enable_cost_and_targeting_before_more_raw_damage(state):
    state['global']['deck'] += [('Corruption', 0), ('Dark Embrace', 0)]
    assert engine_upgrade_value(('Corruption', 0), state) > engine_upgrade_value(('Bash', 0), state)
    assert engine_upgrade_value(('True Grit', 0), state) > engine_upgrade_value(('Carnage', 0), state)
    assert engine_upgrade_value(('Dark Embrace', 1), state) == 0


def test_corruption_preserves_some_defend_fuel_but_removes_curse_first(state):
    before = engine_remove_value(('Defend_R', 0), state)
    state['global']['deck'] += [('Corruption', 0)]
    assert engine_remove_value(('Defend_R', 0), state) < before
    assert engine_remove_value(('Normality', 0), state) > engine_remove_value(('Strike_R', 0), state)
    assert engine_remove_value(('AscendersBane', 0), state) < 0


def test_engine_relics_obey_conflicts_and_status_synergy(state):
    old_choker = engine_relic_value('Velvet Choker', state)
    old_pain = engine_relic_value('Mark of Pain', state)
    state['global']['deck'] += [('Corruption', 0), ('Dark Embrace', 0), ('Offering', 0), ('Evolve', 0)]
    assert engine_relic_value('Velvet Choker', state) < old_choker - 7
    assert engine_relic_value('Mark of Pain', state) > old_pain + 4
    assert engine_relic_value('Runic Dome', state) < 0
    state['global']['relics'] = ['Snecko Eye']
    assert engine_relic_value('Runic Pyramid', state) < 6


def test_reward_mask_and_real_shop_price_are_respected(state):
    state['options']['cards'] = [('Offering', 0), ('Clash', 0)]
    rows, _, _, _ = rank_engine_overworld(state, [A._RCARD + 1, A._RSKIP])
    assert rows[0]['action'] == A._RSKIP
    state['phase'] = 'shop'
    state['options'] = {'shop_cards': [('Offering', 0, 60), ('Clash', 0, 25)]}
    assert rank_engine_overworld(state, [A._SHOP_CARD, A._SHOP_CARD + 1, A._PROCEED])[0][0]['action'] == A._SHOP_CARD
    state['options']['shop_cards'][0] = ('Offering', 0, 999)
    assert rank_engine_overworld(state, [A._SHOP_CARD, A._PROCEED])[0][0]['action'] == A._PROCEED


def test_candidates_do_not_depend_on_offer_order_or_hidden_fields(state):
    before = deepcopy(state)
    expected = engine_card_value(('Offering', 0), state)
    state['options']['cards'] = [('Demon Form', 1)]
    state['global']['deck'].reverse()
    state['future_draw_order'] = object()
    assert engine_card_value(('Offering', 0), state) == expected
    state['global']['deck'].reverse()
    assert state['global'] == before['global']


def request_for(state, legal, seed):
    mask = np.zeros(A.N_ACTIONS, dtype=bool)
    mask[legal] = True
    base = SimpleNamespace(obs_canonical=lambda: state, rmap=None,
        gc=SimpleNamespace(cur_map_node_y=0))
    env = SimpleNamespace(thick=base, action_masks=lambda: mask)
    env.unwrapped = env
    return DecisionRequest(env, {}, {'phase': state['phase']}, seed, 0)


def test_strategy_batches_unknown_fallback_preserves_order_and_traces(state):
    class Fallback:
        name = 'sentinel'
        def decide(self, requests, *, check):
            assert [request.seed for request in requests] == [2, 4]
            return [Decision(A._EVENT + 1, self.name) for _ in requests]
    state['options']['cards'] = [('Carnage', 0)]
    unknown = deepcopy(state)
    unknown['phase'], unknown['options'] = 'event', {'event': 'UNKNOWN_EVENT'}
    before = deepcopy([state, unknown])
    requests = [request_for(state, [A._RCARD, A._RSKIP], 1),
        request_for(unknown, [A._EVENT + 1], 2),
        request_for(state, [A._RCARD, A._RSKIP], 3),
        request_for(unknown, [A._EVENT + 1], 4)]
    traces = []
    strategy = EngineUpperStrategy(Fallback(), trace=traces.append)
    decisions = strategy.decide(requests)
    assert [d.action for d in decisions] == [A._RCARD, A._EVENT + 1, A._RCARD, A._EVENT + 1]
    assert decisions[0].diagnostics['overworld_engine_decisions'] == 1
    assert decisions[1].diagnostics['overworld_fallback_decisions'] == 1
    assert sorted(t['seed'] for t in traces) == [1, 2, 3, 4]
    assert all('engine' in t for t in traces if t['seed'] in (1, 3))
    assert strategy_manifest(strategy)['fallback']['name'] == 'sentinel'
    assert [state, unknown] == before
    assert strategy.decide([]) == []


def test_unknown_events_and_combat_keep_existing_contract(state):
    state['phase'], state['options'] = 'event', {'event': 'UNKNOWN_EVENT'}
    assert rank_engine_overworld(state, [A._EVENT]) is None
    with pytest.raises(ValueError, match='without fallback'):
        EngineUpperStrategy().decide([request_for(state, [A._EVENT], 1)])
    state['phase'] = 'combat'
    with pytest.raises(ValueError, match='combat strategy'):
        rank_engine_overworld(state, [A.END_TURN])


def test_unmodified_event_decisions_equal_baseline(state):
    state['phase'], state['options'] = 'event', {'event': 'Big Fish'}
    legal = [A._EVENT, A._EVENT + 1, A._EVENT + 2]
    rows, original, _, touched = rank_engine_overworld(state, legal)
    assert rows == rank_overworld(state, legal)
    assert rows[0]['action'] == original
    assert not touched


@pytest.mark.parametrize('boss', ['HEXAGHOST', 'SLIME_BOSS', 'DONU_AND_DECA', 'TIME_EATER'])
def test_public_status_boss_enables_one_evolve_without_self_generated_status(state, boss):
    state['global'].update(act=3 if boss in ('DONU_AND_DECA', 'TIME_EATER') else 1,
        floor=45 if boss in ('DONU_AND_DECA', 'TIME_EATER') else 12, boss=boss)
    state['global']['deck'] += [('Carnage', 0), ('Anger', 0)]
    assert engine_profile(state).status_sources == 0
    assert boss_card_adjustment(('Evolve', 0), state) > 0
    assert choose(state, ('Evolve', 0)) == A._RCARD
    # A boss produces temporary status cards, not permanent deck fuel. Do not
    # turn that one fight into a fictitious multi-copy engine commitment.
    state['global']['deck'] += [('Evolve', 0)]
    assert boss_card_adjustment(('Evolve', 0), state) == 0
    assert choose(state, ('Evolve', 0)) == A._RSKIP
    assert 'status' not in engine_profile(state).commitments


@pytest.mark.parametrize('boss', ['THE_GUARDIAN', 'AUTOMATON', 'COLLECTOR', 'CHAMP', 'UNKNOWN'])
def test_ordinary_or_unknown_boss_does_not_invent_status_fuel(state, boss):
    state['global'].update(act=2, floor=29, boss=boss)
    state['global']['deck'] += [('Carnage', 0), ('Anger', 0)]
    assert boss_card_adjustment(('Evolve', 0), state) == 0
    assert choose(state, ('Evolve', 0), ('Fire Breathing', 0)) == A._RSKIP
    # Independently owned status generators still make Evolve useful.
    state['global']['deck'] += [('Power Through', 0), ('Immolate', 0)]
    assert choose(state, ('Evolve', 0)) == A._RCARD


def test_status_boss_prior_waits_for_opening_damage_readiness(state):
    state['global']['boss'] = 'HEXAGHOST'
    assert boss_card_adjustment(('Evolve', 0), state) == 0
    assert choose(state, ('Evolve', 0), ('Carnage', 0)) == A._RCARD + 1
    state['global']['floor'] = 12
    assert boss_card_adjustment(('Evolve', 0), state) == 0
    state['global']['deck'] += [('Carnage', 0), ('Anger', 0)]
    assert boss_card_adjustment(('Evolve', 0), state) > 0


def test_time_eater_status_prior_has_actual_ascension_boundary(state):
    state['global'].update(act=3, floor=45, boss='TIME_EATER', ascension=18)
    assert boss_card_adjustment(('Evolve', 0), state) == 0
    state['global']['ascension'] = 19
    assert boss_card_adjustment(('Evolve', 0), state) > 0


def test_hexaghost_and_champ_missing_scaling_get_access_not_extra_setup(state):
    state['global'].update(floor=13, boss='HEXAGHOST')
    state['global']['deck'] += [('Carnage', 0), ('Anger', 0)]
    assert choose(state, ('Cleave', 0), ('Spot Weakness', 0)) == A._RCARD + 1
    assert boss_card_adjustment(('Demon Form', 0), state) == 0
    state['global']['deck'] += [('Inflame', 0)]
    assert boss_card_adjustment(('Spot Weakness', 0), state) == 0
    state['global'].update(act=2, floor=29, boss='CHAMP')
    assert boss_card_adjustment(('Limit Break', 0), state) > 0
    state['global']['deck'] = [('Strike_R', 0)] * 5
    assert boss_card_adjustment(('Limit Break', 0), state) == 0
    assert choose(state, ('Limit Break', 0)) == A._RSKIP


def test_guardian_block_gap_and_awakened_selective_power_caution(state):
    state['global'].update(floor=13, boss='THE_GUARDIAN')
    state['global']['deck'] += [('Carnage', 0), ('Anger', 0)]
    assert boss_card_adjustment(('Shrug It Off', 0), state) > 0
    state['global']['deck'] += [('Shrug It Off', 0), ('Power Through', 0)]
    assert boss_card_adjustment(('Shrug It Off', 0), state) == 0
    state['global'].update(act=3, floor=45, boss='AWAKENED_ONE')
    assert boss_card_adjustment(('Brutality', 0), state) < 0
    assert boss_card_adjustment(('Metallicize', 0), state) < 0
    for core in ('Corruption', 'Dark Embrace', 'Feel No Pain', 'Inflame'):
        assert boss_card_adjustment((core, 0), state) == 0


def test_owned_evolve_upgrade_remains_useful_against_public_status_boss(state):
    state['global'].update(floor=13, boss='HEXAGHOST')
    state['global']['deck'] += [('Carnage', 0), ('Anger', 0), ('Evolve', 0)]
    needed = engine_upgrade_value(('Evolve', 0), state)
    state['global']['boss'] = 'THE_GUARDIAN'
    assert engine_upgrade_value(('Evolve', 0), state) < needed
    state['global']['deck'] += [('Power Through', 0)]
    assert engine_upgrade_value(('Evolve', 0), state) == needed


@pytest.mark.parametrize('block', ['True Grit', 'Ghostly Armor'])
def test_compact_deck_does_not_skip_needed_bridge_block_before_first_boss(state, block):
    state['global'].update(floor=12, boss='HEXAGHOST')
    assert boss_card_adjustment((block, 0), state) == 0  # opening damage missing
    state['global']['deck'] += [('Anger', 0), ('Carnage', 0)]
    assert choose(state, (block, 0)) == A._RCARD
    state['global']['deck'] += [('Power Through', 0), ('Shrug It Off', 0)]
    assert boss_card_adjustment((block, 0), state) == 0
    assert choose(state, (block, 0)) == A._RSKIP
