"""A narrow Act-2 card-reward experiment, composed with overworld rules v1.

The feature is an exact expectation over *hypothetical uniform five-card hands*
from the public, unordered deck. It is not a simulation, an opening-hand model,
or expected HP loss. Only affordable direct reusable block is represented;
draw, debuffs, damage, setup, Frail and enemy growth retain their v1 priors.
"""
from __future__ import annotations

from collections import Counter
from functools import lru_cache
from math import comb

import numpy as np

from env.core import action_space as A
from env.core.ironclad_cards import card_numbers
from rl.evaluation.baselines.overworld_rules import (
    OverworldRuleStrategy, _item_key, rank_overworld,
)
from rl.evaluation.strategies import Decision

OVERWORLD_DEMAND_VERSION = 'ironclad-overworld-demand-v1'

# Power Through's Wounds, Impervious's one use, and True Grit/Second Wind's
# hand-card consumption are not represented as reusable flat block. In
# particular five True Grits with four energy cannot actually all be played.
DIRECT_BLOCK = frozenset(('Defend_R', 'Armaments', 'Iron Wave', 'Shrug It Off',
    'Flame Barrier', 'Ghostly Armor', 'Sentinel'))
UNMODELED_ENGINES = frozenset(('Corruption', 'Feel No Pain', 'Dark Embrace',
    'Barricade', 'Entrench', 'Second Wind', 'Rage', 'Metallicize', 'Juggernaut'))
UNMODELED_RELICS = frozenset(('SNECKOEYE', 'RUNICPYRAMID', 'CALIPERS',
    'ORICHALCUM', 'DEADBRANCH', 'KUNAI'))
ENERGY_RELICS = frozenset(('BUSTEDCROWN', 'COFFEEDRIPPER', 'CURSEDKEY',
    'ECTOPLASM', 'FUSIONHAMMER', 'PHILOSOPHERSSTONE', 'RUNICDOME',
    'SLAVERSCOLLAR', 'SOZU', 'VELVETCHOKER', 'MARKOFPAIN'))


def demand_profile(state):
    """Public reference attack, not the boss's total incoming damage forecast.

    Native MonsterSpecific.cpp: Automaton Flail 7/8 x2, Champ Heavy Slash
    16/18, Collector Fireball 18/21 (A4 breakpoint). Adds, Strength, Weak,
    Vulnerable, Hyper Beam and Execute are outside this one-component feature.
    """
    g = state['global']
    boss = _item_key(g.get('boss', ''))
    # Canonical global.boss is the encounter ID, not the monster ID. Accept
    # bridge display aliases too, without consulting future native encounters.
    boss = {'BRONZEAUTOMATON': 'AUTOMATON', 'THECHAMP': 'CHAMP',
            'THECOLLECTOR': 'COLLECTOR'}.get(boss, boss)
    if g.get('act') != 2 or boss not in ('AUTOMATON', 'CHAMP', 'COLLECTOR'):
        return None
    deck = tuple((c[0], int(c[1])) for c in g.get('deck', ()))
    if not deck or any(c[0] in UNMODELED_ENGINES for c in deck):
        return None
    relics = {_item_key(r) for r in g.get('relics', ())}
    if relics & UNMODELED_RELICS:
        return None
    high = int(g.get('ascension', 0)) >= 4
    reference = {'AUTOMATON': (14, 16), 'CHAMP': (16, 18),
                 'COLLECTOR': (18, 21)}[boss][high]
    # Permanent energy plus Slaver's Collar's boss-only energy; reserve one
    # point for offense. Temporary/conditional energy is not anticipated.
    budget = 2 + len(relics & ENERGY_RELICS)
    dexterity = int('ODDLYSMOOTHSTONE' in relics)
    return dict(boss=boss, reference=reference, energy_budget=budget,
                dexterity=dexterity)


@lru_cache(maxsize=8192)
def hand_block_coverage(deck, energy_budget, reference, dexterity=0):
    """E[min(reference, max affordable direct block)] in a uniform 5-card set.

    Non-block cards remain in the draw denominator. Equal cost/block cards are
    grouped; binomial multiplicities give exact without-replacement weights.
    Dynamic side effects are not simulated. More than 50,000 partial grouped
    hands returns None and delegates to v1, imposing a deterministic work bound.
    """
    groups = Counter()
    for cid, upgrade in deck:
        if cid in DIRECT_BLOCK:
            cost, _, block = card_numbers(cid, upgrade)
            groups[(cost, block + dexterity)] += 1
    size, hand = len(deck), min(5, len(deck))
    if not size:
        return 0.0
    filler = size - sum(groups.values())
    groups = sorted(groups.items())
    denominator = comb(size, hand)
    total = 0.0
    visited = 0

    def visit(index, picked, ways, best):
        nonlocal total, visited
        visited += 1
        if visited > 50_000:
            raise OverflowError
        if index == len(groups):
            other = hand - picked
            if 0 <= other <= filler:
                total += ways * comb(filler, other) * min(reference, max(best))
            return
        (cost, block), available = groups[index]
        values = list(best)
        for count in range(min(available, hand - picked) + 1):
            visit(index + 1, picked + count, ways * comb(available, count), values)
            # 0/1 knapsack: each copy can contribute once, with total cost
            # constrained to the stated block budget.
            for energy in range(energy_budget, cost - 1, -1):
                values[energy] = max(values[energy], values[energy - cost] + block)

    try:
        visit(0, 0, 1, [0] * (energy_budget + 1))
    except OverflowError:
        return None
    return total / denominator


def reward_adjustment(card, state, *, profile=None):
    """Candidate-independent marginal readiness bonus; never changes v1 helpers."""
    profile = demand_profile(state) if profile is None else profile
    if profile is None or card[0] not in DIRECT_BLOCK:
        return 0.0, None
    deck = tuple(sorted((c[0], int(c[1])) for c in state['global']['deck']))
    args = (profile['energy_budget'], profile['reference'], profile['dexterity'])
    before = hand_block_coverage(deck, *args)
    after = hand_block_coverage(tuple(sorted((*deck, (card[0], int(card[1]))))), *args)
    if before is None or after is None:
        return 0.0, {'abstention': 'hand_enumeration_budget'}
    # This score scale is a declared heuristic, not learned/calibrated utility:
    # one additional point of capped block coverage -> one v1 preference point,
    # at most four. Non-block cards keep v1 values; no blanket dilution penalty.
    bonus = min(4.0, max(0.0, after - before))
    return bonus, dict(before=before, after=after, bonus=bonus, **profile)


def rank_demand_rewards(state, legal):
    """Return a v1 reward ranking with only the bounded readiness bonus added."""
    profile = demand_profile(state)
    if state['phase'] != 'combat_reward' or profile is None:
        return None
    if not any(A.segment_of(a)[0] == 'REWARD_CARD' for a in legal):
        return None
    ranked = rank_overworld(state, legal)
    if ranked is None:
        return None
    original_action = ranked[0]['action']
    for row in ranked:
        kind, index = A.segment_of(row['action'])
        if kind == 'REWARD_CARD':
            bonus, detail = reward_adjustment(state['options']['cards'][index], state,
                                              profile=profile)
            row['v1_score'] = row['score']
            row['score'] += bonus
            row['demand'] = detail
    ranked.sort(key=lambda row: (-row['score'], row['action']))
    return ranked, original_action


class EnemyDemandUpperStrategy:
    """Drop-in DecisionStrategy; use PhaseStrategy for combat and model fallback."""
    name = 'overworld-demand'
    version = OVERWORLD_DEMAND_VERSION

    def __init__(self, fallback=None, *, trace=None):
        self.fallback, self.trace = fallback, trace
        self._base = OverworldRuleStrategy(fallback, trace=trace)

    def decide(self, requests, *, check=lambda: None):
        result, delegated, indices = [None] * len(requests), [], []
        for index, request in enumerate(requests):
            check()
            state = request.env.unwrapped.thick.obs_canonical()
            ranked = rank_demand_rewards(state, np.flatnonzero(request.env.action_masks()))
            if ranked is None:
                delegated.append(request)
                indices.append(index)
                continue
            rows, original = ranked
            action = rows[0]['action']
            result[index] = Decision(action, self.name, {
                'overworld_rule_decisions': 1, 'overworld_demand_decisions': 1,
                'overworld_demand_changed_decisions': int(action != original)})
            if self.trace:
                self.trace(dict(kind='overworld_decision', seed=request.seed,
                    step=request.step, phase=state['phase'], version=self.version,
                    ranked=rows, state=state, action=action, source=self.name,
                    v1_action=original))
        if delegated:
            decisions = self._base.decide(delegated, check=check)
            for index, decision in zip(indices, decisions):
                result[index] = decision
        return result
