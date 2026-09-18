"""Opt-in boss-relic payment-capacity prior; no rollout or future-state access.

Uniform five-card sets are hypothetical later-cycle samples, not predicted opening
hands. We maximize existing v1 card-prior utility under a printed-cost budget;
this is an energy-only relaxation, not a feasible combat plan or HP-loss model.
Draw effects are valued by v1, never materialized as free extra sampled cards.
"""
from collections import Counter
from functools import lru_cache
from math import comb

import numpy as np

from env.core import action_space as A
from env.core.ironclad_cards import IRONCLAD_CARDS
from rl.evaluation.baselines.overworld_rules import (
    CURSES, DRAW, OverworldRuleStrategy, _item_key, card_value, rank_overworld,
)
from rl.evaluation.strategies import Decision

OVERWORLD_ENERGY_VERSION = 'ironclad-overworld-energy-v1'
ALWAYS_ENERGY = frozenset(('CURSEDKEY', 'SOZU', 'COFFEEDRIPPER', 'FUSIONHAMMER',
    'PHILOSOPHERSSTONE', 'RUNICDOME', 'VELVETCHOKER', 'MARKOFPAIN', 'ECTOPLASM', 'BUSTEDCROWN'))
SUPPORTED = (ALWAYS_ENERGY - {'RUNICDOME', 'VELVETCHOKER', 'BUSTEDCROWN'}) | {'SLAVERSCOLLAR'}
# Temporary energy/cost conversion makes printed payment capacity misleading.
UNMODELED_CARDS = frozenset(('Corruption', 'Offering', 'Bloodletting', 'Seeing Red',
    'Berserk', 'Sentinel', 'Blood For Blood'))
UNMODELED_RELICS = frozenset(('SNECKOEYE', 'RUNICPYRAMID', 'ICECREAM', 'SUNDIAL',
    'HAPPYFLOWER', 'MUMMIFIEDHAND', 'NECRONOMICON', 'VELVETCHOKER'))


@lru_cache(maxsize=4096)
def affordable_prior_curve(cards, max_energy):
    """Exact uniform-5 expectation of the stated cost-only card utility model.

    (cost, value) cards with zero value are filler. Nonnegative costs only;
    unplayable cards are represented as (0, 0), without returning energy.
    Return utilities for budgets0..max_energy, or None at the work bound.
    """
    if max_energy < 0 or any(cost < 0 or value < 0 for cost, value in cards):
        raise ValueError('costs, utilities and energy must be nonnegative')
    if not cards:
        return tuple(0.0 for _ in range(max_energy + 1))
    groups = Counter((cost, value) for cost, value in cards if value > 0)
    filler = len(cards) - sum(groups.values())
    groups = sorted(groups.items())
    hand = min(5, len(cards))
    denominator = comb(len(cards), hand)
    total, visits = [0.0] * (max_energy + 1), 0

    def visit(index, picked, ways, best):
        nonlocal visits
        visits += 1
        if visits > 100_000:
            raise OverflowError
        if index == len(groups):
            other = hand - picked
            if 0 <= other <= filler:
                weight = ways * comb(filler, other)
                for energy, value in enumerate(best):
                    total[energy] += weight * value
            return
        (cost, value), available = groups[index]
        values = list(best)
        for count in range(min(available, hand - picked) + 1):
            visit(index + 1, picked + count, ways * comb(available, count), values)
            for energy in range(max_energy, cost - 1, -1):
                values[energy] = max(values[energy], values[energy - cost] + value)

    try:
        visit(0, 0, 1, [0.0] * (max_energy + 1))
    except OverflowError:
        return None
    return tuple(value / denominator for value in total)


def energy_profile(state):
    g = state['global']
    deck = g.get('deck', ())
    relics = {_item_key(r) for r in g.get('relics', ())}
    if not deck or relics & UNMODELED_RELICS or any(c in UNMODELED_CARDS for c, _ in deck):
        return None
    cards = []
    for card in deck:
        cid, up = card[:2]
        info = IRONCLAD_CARDS.get(cid)
        # Known curses/status are zero-valued filler; unsupported colorless or
        # generated cards cause abstention rather than inventing their costs.
        if info is None:
            if cid in CURSES or cid in ('Wound', 'Burn', 'Dazed'):
                cards.append((0, 0.0))
                continue
            return None
        cost = info.cost[bool(up)]
        if cost < 0:  # X costs require a separate resource model.
            return None
        cards.append((cost, max(0.0, card_value(card, state, adding=False))))
    if 'MARKOFPAIN' in relics:
        # The permanent deck omits battle-generated Wounds. Apply the same
        # post-insertion distribution to an already owned Mark of Pain.
        cards.extend(((0, 0.0), (0, 0.0)))
    return dict(cards=tuple(sorted(cards)), energy=3 + len(relics & ALWAYS_ENERGY),
        collar=int('SLAVERSCOLLAR' in relics), relics=relics,
        draw_cards=sum(c in DRAW for c, _ in deck))


def rank_energy_relics(state, legal):
    if state['phase'] != 'boss_relic':
        return None
    profile = energy_profile(state)
    if profile is None:
        return None
    ranked = rank_overworld(state, legal)
    if ranked is None:
        return None
    original = ranked[0]['action']
    energy = profile['energy']
    baseline = affordable_prior_curve(profile['cards'], energy + profile['collar'] + 1)
    if baseline is None:
        return None
    supported = False
    for row in ranked:
        kind, index = A.segment_of(row['action'])
        if kind != 'BOSS_RELIC':
            continue
        relic = _item_key(state['global']['boss_relics'][index])
        if relic not in SUPPORTED:
            continue
        cards = profile['cards']
        if relic == 'MARKOFPAIN':
            cards = tuple(sorted((*cards, (0, 0.0), (0, 0.0))))
        after = affordable_prior_curve(cards, energy + profile['collar'] + 1)
        if after is None:
            continue
        # Equal hallway and elite/boss contexts is an explicit static prior.
        # Collar only supplies its bonus in the latter context.
        regular_gain = int(relic != 'SLAVERSCOLLAR')
        delta = .5 * (after[energy + regular_gain] - baseline[energy])
        delta += .5 * (after[energy + profile['collar'] + 1] - baseline[energy + profile['collar']])
        correction = max(-4.0, min(4.0, .65 * delta))
        row['v1_score'] = row['score']
        row['score'] += correction
        row['energy'] = dict(base_energy=energy, existing_collar=profile['collar'],
            marginal_prior=delta, correction=correction, draw_cards=profile['draw_cards'],
            wound_filler=2 if relic == 'MARKOFPAIN' else 0,
            medical_kit='MEDICALKIT' in profile['relics'])
        supported = True
    if not supported:
        return None
    ranked.sort(key=lambda row: (-row['score'], row['action']))
    return ranked, original


class EnergyAwareUpperStrategy:
    name = 'overworld-energy'
    version = OVERWORLD_ENERGY_VERSION

    def __init__(self, fallback=None, *, trace=None):
        self.fallback, self.trace = fallback, trace
        self._base = OverworldRuleStrategy(fallback, trace=trace)

    def manifest(self):
        return dict(scope='boss-relic-only', base_version=self._base.version,
            feature='uniform-five-card-cost-only-v1-prior-capacity',
            draw_model='v1-prior-no-extra-sampled-cards',
            side_effects='preserved-v1-priors-plus-mark-of-pain-draw-dilution',
            energy_contexts='half-hallway-half-elite-or-boss')

    def decide(self, requests, *, check=lambda: None):
        result, delegated, indices = [None] * len(requests), [], []
        for index, request in enumerate(requests):
            check()
            state = request.env.unwrapped.thick.obs_canonical()
            ranked = rank_energy_relics(state, np.flatnonzero(request.env.action_masks()))
            if ranked is None:
                delegated.append(request)
                indices.append(index)
                continue
            rows, original = ranked
            action = rows[0]['action']
            result[index] = Decision(action, self.name, dict(overworld_rule_decisions=1,
                overworld_energy_decisions=1, overworld_energy_changed_decisions=int(action != original)))
            if self.trace:
                self.trace(dict(kind='overworld_decision', seed=request.seed, step=request.step,
                    phase=state['phase'], version=self.version, ranked=rows, state=state,
                    action=action, source=self.name, v1_action=original))
        if delegated:
            decisions = self._base.decide(delegated, check=check)
            for index, request, decision in zip(indices, delegated, decisions):
                if request.info.get('phase') == 'boss_relic':
                    decision = Decision(decision.action, decision.source,
                        dict(decision.diagnostics, overworld_energy_abstentions=1),
                        native_action=decision.native_action)
                result[index] = decision
        return result
