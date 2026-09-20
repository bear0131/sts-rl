"""A20 challenger: repair encounter coverage before buying speculative scaling.

All numbers are declared human priors, not calibrated damage/HP forecasts. The
map planner propagates a coarse *risk budget* over visible room types; it never
queries a future encounter, reward, event outcome or draw order. Existing v1
event semantics and explicit model fallback remain authoritative.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache

import numpy as np

from env.core import action_space as A
from env.core.ironclad_cards import IRONCLAD_CARDS
from env.core.spiremap import MAX_Y
from rl.evaluation.baselines.overworld_potion_shop import EMPTY_SLOTS, potion_preference
from rl.evaluation.baselines.overworld_rules import (
    CARD_VALUE, CURSES, DRAW, EXHAUST, STRENGTH, OverworldRuleStrategy,
    _item_key, deck_counts, has_relic, rank_overworld, relic_value,
    relic_strength_sources, remove_value, upgrade_value,
)
from rl.evaluation.strategies import Decision

OVERWORLD_SURVIVAL_VERSION = 'ironclad-overworld-survival-v1.1-relic-contracts'
OVERWORLD_GROWTH_VERSION = 'ironclad-overworld-growth-v1.1-relic-contracts'
OVERWORLD_READINESS_VERSION = 'ironclad-overworld-readiness-v1.1-relic-contracts'
FRONTLOAD = frozenset(('Anger', 'Carnage', 'Hemokinesis', 'Bludgeon',
    'Immolate', 'Fiend Fire', 'Perfected Strike', 'Pommel Strike', 'Twin Strike',
    'Headbutt', 'Uppercut', 'Clothesline', 'Wild Strike', 'Cleave', 'Whirlwind'))
FAST_AOE = frozenset(('Cleave', 'Whirlwind', 'Immolate'))
GOOD_BLOCK = frozenset(('Shrug It Off', 'Flame Barrier', 'Ghostly Armor',
    'Power Through', 'Impervious', 'Second Wind', 'True Grit'))
WEAK = frozenset(('Shockwave', 'Uppercut', 'Clothesline', 'Intimidate'))
ENGINE = frozenset(('Barricade', 'Entrench', 'Dark Embrace', 'Corruption',
    'Demon Form', 'Limit Break', 'Rupture', 'Juggernaut', 'Dual Wield'))
ENERGY_RELICS = frozenset(('CURSEDKEY', 'SOZU', 'COFFEEDRIPPER', 'FUSIONHAMMER',
    'PHILOSOPHERSSTONE', 'RUNICDOME', 'VELVETCHOKER', 'MARKOFPAIN',
    'ECTOPLASM', 'BUSTEDCROWN'))


@dataclass(frozen=True)
class UpperProfile:
    name: str
    elite_growth_bonus: float
    smith_multiplier: float
    rest_targets: tuple[float, float, float]
    boss_rest_targets: tuple[float, float, float]
    elite_policy: str = 'standard'


SURVIVAL_PROFILE = UpperProfile('survival', 0.0, 1.0, (.68, .76, .76), (.82, .82, .82))
GROWTH_PROFILE = UpperProfile('growth', 6.0, 1.4, (.46, .56, .62), (.60, .70, .78))
READINESS_PROFILE = UpperProfile('readiness', 0.0, 1.0, (.68, .76, .76),
                                (.82, .82, .82), elite_policy='readiness')


@dataclass(frozen=True)
class Coverage:
    frontload: int
    aoe: int
    defense: int
    weak: int
    strength: int
    exhaust: int
    draw: int
    energy: int


def coverage(state):
    counts = deck_counts(state)
    relics = {_item_key(r) for r in state['global'].get('relics', ())}
    return Coverage(
        sum(counts[c] for c in FRONTLOAD), sum(counts[c] for c in FAST_AOE),
        sum(counts[c] for c in GOOD_BLOCK), sum(counts[c] for c in WEAK),
        sum(counts[c] for c in STRENGTH) + relic_strength_sources(state),
        sum(counts[c] for c in EXHAUST), sum(counts[c] for c in DRAW),
        3 + len(relics & ENERGY_RELICS),
    )


def elite_ready(state, *, potion_reserves=None):
    """A deliberately conservative unlock for growth's extra elite reward.

This only inspects the actual public deck/potions, not an unseen elite identity.
The route approximation consumes one available reserve per projected elite, so
one potion cannot unlock an arbitrary chain of future elite fights.
"""
    g, p, counts = state['global'], coverage(state), deck_counts(state)
    if potion_reserves is None:
        potion_reserves = sum(bool(potion_preference(x, state)) for x in g.get('potions', ()))
    act = int(g.get('act', 1))
    if act == 1:
        upgraded_nuke = any(c in ('Carnage', 'Immolate', 'Bludgeon', 'Fiend Fire') and up
                            for c, up in g.get('deck', ()))
        return p.frontload >= 3 or (p.frontload >= 2 and (upgraded_nuke or potion_reserves > 0))
    if act == 2:
        return p.frontload >= 3 and p.aoe >= 1 and p.defense >= 2 and (p.weak > 0 or potion_reserves > 0)
    engine = (counts['Corruption'] and counts['Feel No Pain']) or (
        counts['Barricade'] and (counts['Impervious'] or counts['Power Through']))
    return p.aoe >= 1 and p.defense >= 3 and (p.strength > 0 or bool(engine))


def survival_card_value(card, state, *, adding=True):
    """Marginal acquisition priority on a zero-is-skip scale.

Coverage deficits dominate speculative combinations early. Once covered, extra
attacks lose priority and cards that improve draw/block/scaling can enter. No
archetype is committed to solely because one component happens to be present.
"""
    cid, up = card[:2]
    if cid in CURSES:
        return -30.0
    g, c, p = state['global'], deck_counts(state), coverage(state)
    act, floor = int(g.get('act', 1)), int(g.get('floor', 1))
    duplicate = c[cid] if adding else max(0, c[cid] - 1)
    value = CARD_VALUE.get(cid, 2.5) - (5.4 if act == 1 else 6.0)
    value -= duplicate * (3.5 if cid in ENGINE or cid in ('Anger', 'Battle Trance', 'Armaments') else 1.7)
    if cid in FRONTLOAD:
        if act == 1:
            value += (3.2, 2.0, 1.0, 0.0)[min(p.frontload, 3)]
        elif cid not in FAST_AOE:
            value -= 1.2 + .5 * max(0, p.frontload - 4)
    if cid in FAST_AOE:
        value += (3.8 if act >= 2 else 2.4) if p.aoe == 0 else -1.3 * p.aoe
        if cid == 'Immolate':
            value += 1.5  # One card immediately clears many multi-enemy boards.
    if cid in GOOD_BLOCK:
        target = 2 if act == 1 else 4
        value += 1.2 * max(0, target - p.defense)
        if act == 1 and floor <= 8 and p.frontload < 2:
            value -= 2.0
    if cid in WEAK:
        value += (2.0 if act >= 2 else .8) if not p.weak else -.8 * p.weak
    if cid == 'Disarm':
        value += (2 if act >= 2 else .5) - 1.2 * duplicate
    if cid in DRAW:
        value += min(2.5, max(0, len(g.get('deck', ())) - 14) / 5)
    if cid in STRENGTH:
        value += (3.0 if act >= 2 else 1.0) if not p.strength else -1.5 * p.strength
    if cid in ENGINE and act == 1 and p.frontload < 3:
        value -= 3.5
    if cid in ('Heavy Blade', 'Sword Boomerang', 'Pummel', 'Reaper'):
        value += min(4.5, p.strength * 2)
    if cid == 'Reaper':
        value += 1.0 + 2.0 * (1 - g['hp'] / max(1, g['max_hp']))
    if cid == 'Limit Break':
        value += 7 if p.strength else -12
    if cid == 'Demon Form':
        value += 2 if act >= 2 and not p.strength else -2
        value -= 1.5 if p.energy == 3 else 0
    if cid in ('Feel No Pain', 'Dark Embrace'):
        value += min(6, p.exhaust * 1.1 + 2 * c['Corruption'])
        if cid == 'Feel No Pain':
            value += .8  # Ascender's Bane and enemy statuses also exhaust.
    if cid == 'Corruption':
        skills = sum(n for name, n in c.items()
                     if name in IRONCLAD_CARDS and IRONCLAD_CARDS[name].kind == 'skill')
        value += min(3, max(0, skills - 5) * .6) + 2 * min(1, c['Feel No Pain'] + c['Dark Embrace'])
        value += 5 if has_relic(state, 'DEAD_BRANCH') else 0
    if cid in EXHAUST:
        value += min(3, c['Feel No Pain'] + c['Dark Embrace'])
    if cid == 'Second Wind':
        value += 2 * min(1, c['Power Through']) + min(2, c['Feel No Pain'])
    if cid == 'Power Through':
        value += 2 * min(1, c['Second Wind'] + c['Evolve'] + c['Corruption'])
    if cid == 'Barricade':
        value += 2 * min(2, c['Impervious'] + c['Power Through']) + 2 * min(1, c['Entrench'])
    if cid == 'Entrench':
        value += 7 if c['Barricade'] or has_relic(state, 'CALIPERS') else -5
    if cid == 'Body Slam':
        value += min(6, c['Barricade'] * 3 + c['Power Through'] * 2 + c['Impervious'])
    if cid == 'Rupture':
        value += min(7, 2 * sum(c[x] for x in ('Combust', 'Brutality', 'Hemokinesis', 'Offering')))
    if cid in ('Evolve', 'Fire Breathing'):
        value += min(4, 1.7 * sum(c[x] for x in ('Power Through', 'Wild Strike', 'Reckless Charge')))
        value += 1 if act == 1 and floor >= 7 else 0
    if cid == 'Perfected Strike':
        value += .8 * (sum(n for name, n in c.items() if 'Strike' in name) - 6)
    if cid in ('Wild Strike', 'Reckless Charge') and not (c['Evolve'] or c['Fire Breathing']):
        value -= 1.5 if act > 1 else .5
    if cid == 'Feed':
        value -= max(0, floor - 10) * .11
    if cid == 'Clash':
        value = -20  # A20's permanent Ascender's Bane already breaks reliability.
    if cid in ('Strike_R', 'Defend_R', 'Bash'):
        value -= 5
    if up:
        value += min(2.5, upgrade_value((cid, 0), state) / 4)
    if adding and len(g.get('deck', ())) > 25 and cid not in DRAW:
        value -= min(3, .25 * (len(g['deck']) - 25))
    return float(value)


def survival_upgrade_value(card, state):
    value = upgrade_value(card, state)
    if card[1] and card[0] != 'Searing Blow':
        return 0.0
    cid, p, act = card[0], coverage(state), state['global'].get('act', 1)
    if cid in ('Carnage', 'Hemokinesis', 'Immolate', 'Whirlwind'):
        value += 4 if act == 1 or p.aoe <= 1 else 1
    if cid == 'Bash' and act == 1 and p.frontload <= 2:
        value += 2
    if cid == 'Disarm' and act >= 2:
        value += 4
    return value


def survival_route_scores(state, choices, rmap, current_y, *, profile=SURVIVAL_PROFILE):
    """Discounted visible routes with consecutive-fight and recovery awareness.

Risk costs below are coarse HP-budget priors, deliberately not forecasts. A
healthy prepared deck can take an elite; the same rooms without intervening
rest become unattractive as the projected budget falls. Replan every room.
"""
    g, p = state['global'], coverage(state)
    act, max_hp = int(g.get('act', 1)), max(1, int(g['max_hp']))
    can_heal = not has_relic(state, 'COFFEE_DRIPPER', 'MARK_OF_THE_BLOOM')
    smith = 0 if has_relic(state, 'FUSION_HAMMER') else max(
        (survival_upgrade_value(c, state) for c in g.get('deck', ())), default=0)
    missing = max(0, 3 - p.frontload) if act == 1 else max(0, 1 - p.aoe) + max(0, 3 - p.defense)
    potion_count = sum(bool(potion_preference(x, state)) for x in g.get('potions', ()) if x not in EMPTY_SLOTS)
    hall_cost = (5 if act == 1 else 9) + 2 * missing
    elite_cost = (17 if act == 1 else 24) + 3 * missing - min(6, 3 * potion_count)
    heal_after_fight = 6 if has_relic(state, 'BURNING_BLOOD') else 12 if has_relic(state, 'BLACK_BLOOD') else 0
    if not can_heal and has_relic(state, 'MARK_OF_THE_BLOOM'):
        heal_after_fight = 0

    @lru_cache(None)
    def visit(x, y, hp_bin, reserves):
        if y > MAX_Y:
            return 0.0
        hp, room = hp_bin * 4.0, rmap.room_type(x, y)
        own, after = 0.0, hp
        next_reserves = reserves
        if room in ('MONSTER', 'ELITE'):
            cost = elite_cost if room == 'ELITE' else hall_cost
            own = (7.0 if act == 1 else 5.0) if room == 'ELITE' else (2.7 if act == 1 and p.frontload < 4 else .2)
            risk_buffer = .25
            if profile.elite_growth_bonus or profile.elite_policy == 'readiness':
                if room == 'ELITE':
                    cost += min(6, 3 * potion_count) - min(6, 3 * reserves)
                    ready = elite_ready(state, potion_reserves=reserves)
                    if profile.elite_policy == 'readiness':
                        hp_floor = (.60, .68, .70)[min(max(act, 1), 3) - 1]
                        if not ready or hp < max(hp_floor * max_hp, cost + .25 * max_hp):
                            own, risk_buffer = 1.0, .35
                    elif ready and hp >= max(.45 * max_hp, cost + .15 * max_hp):
                        own += profile.elite_growth_bonus
                        own += 2 if has_relic(state, 'BLACK_STAR') else 0
                        cost -= 5 if has_relic(state, 'PRESERVED_INSECT') else 0
                        risk_buffer = .15
                    next_reserves = max(0, reserves - 1)
                elif profile.elite_growth_bonus and act == 1 and p.frontload < 5:
                    own += 1.2  # More card rewards before an elite, not only HP conservation.
            own -= .10 * cost + 12 * max(0.0, (cost + risk_buffer * max_hp - hp) / max_hp)
            if hp <= cost:
                own -= 20  # An earlier reward cannot compensate likely death.
            after = max(0, hp - cost) + heal_after_fight
        elif room == 'REST':
            heal = min(.3 * max_hp, max_hp - hp) if can_heal else 0
            smith_score = smith * profile.smith_multiplier
            own = max(smith_score / 5, heal / 5)
            after = hp + heal if heal >= smith_score else hp
        elif room == 'SHOP':
            own = min(5, max(-2, (g.get('gold', 0) - 70) / 45))
        elif room == 'EVENT':
            own = 1.6 if act > 1 else 1.0
        elif room == 'TREASURE':
            own = 3.0
        children = rmap.reachable(x, y) if y < MAX_Y else ()
        next_hp = int(min(max_hp, after) // 4)
        future = max((visit(child, y + 1, next_hp, next_reserves) for child in children), default=0)
        return own + .82 * future

    return {action: visit(x, max(0, current_y + 1), int(g['hp'] // 4), potion_count) for x, action in choices}


def rank_survival(state, legal, *, rmap=None, current_y=-1, profile=SURVIVAL_PROFILE):
    rows = rank_overworld(state, legal, rmap=rmap, current_y=current_y)
    if rows is None:
        return None
    g, opts, p = state['global'], state.get('options', {}), coverage(state)
    hp_ratio, act = g['hp'] / max(1, g['max_hp']), int(g.get('act', 1))
    context = opts.get('select', {}).get('context')
    routes = {}
    if state['phase'] == 'map':
        choices = [(i, a) for a in legal for kind, i in [A.segment_of(a)] if kind == 'MAP_NODE']
        routes = survival_route_scores(state, choices, rmap, current_y, profile=profile)
    for row in rows:
        kind, i = A.segment_of(row['action'])
        row['v1_score'] = row['score']
        if kind == 'REWARD_CARD':
            row['score'] = survival_card_value(opts['cards'][i], state)
            row['reason'] = 'coverage:' + opts['cards'][i][0]
        elif kind == 'PICK_CARD' and context in ('OBTAIN', 'DUPLICATE', 'BOTTLE'):
            row['score'] = survival_card_value(opts['cards'][i], state, adding=context != 'BOTTLE')
        elif kind == 'PICK_CARD' and context == 'UPGRADE':
            row['score'] = survival_upgrade_value(opts['cards'][i], state)
        elif kind == 'MAP_NODE':
            row['score'], row['reason'] = routes[row['action']], 'public-route-risk-budget'
        elif kind == 'CAMPFIRE':
            name = A._CAMP_NAME_BY_IDX[i]
            if name == 'rest':
                missing = max(0, g['max_hp'] - g['hp'])
                near_boss = current_y >= 13
                targets = profile.boss_rest_targets if near_boss else profile.rest_targets
                target = targets[min(max(act, 1), 3) - 1]
                row['score'] = min(missing, .3 * g['max_hp']) * (.85 if hp_ratio < target else .22)
            elif name == 'smith':
                row['score'] = profile.smith_multiplier * max(
                    (survival_upgrade_value(c, state) for c in g.get('deck', ())), default=0)
        elif kind == 'SHOP_CARD':
            cid, up, price = opts['shop_cards'][i]
            row['score'] = survival_card_value((cid, up), state) * 2 - price / 25
        elif kind == 'SHOP_POTION':
            potion, price = opts['shop_potions'][i]
            held = g.get('potions', ())
            can_obtain = any(x in EMPTY_SLOTS for x in held) and not has_relic(state, 'SOZU')
            pref = potion_preference(potion, state)
            reserves = sum(x not in EMPTY_SLOTS and x != 'INVALID' for x in held)
            row['score'] = -100
            if can_obtain and pref and price >= 0:
                priority, role = pref
                row['score'] = priority + (1.5 if act >= 2 else 0) - 2.5 * reserves - price / 22
                row['reason'] = f'reserve:{role}:{potion}'
        elif kind == 'SHOP_REMOVE':
            removable_curse = any(c in CURSES and remove_value((c, 0), state) > 0 for c, _ in g.get('deck', ()))
            if act == 1 and p.frontload < 2 and not removable_curse:
                row['score'] = -1  # Buy damage before spending most gold on a Strike.
        elif kind == 'BOSS_RELIC':
            relic = g['boss_relics'][i]
            key = _item_key(relic)
            value = relic_value(relic, state)
            if key == 'SOZU':
                value -= 4  # Two A20 potion slots still carry large encounter coverage.
            elif key == 'PHILOSOPHERSSTONE':
                value -= 3 if act == 1 else 1.5
            elif key == 'MARKOFPAIN':
                value -= 2.5 if not deck_counts(state)['Evolve'] else -3
            elif key == 'COFFEEDRIPPER' and not (deck_counts(state)['Reaper'] or has_relic(state, 'MEAT_ON_THE_BONE', 'ETERNAL_FEATHER', 'BLACK_BLOOD')):
                value -= 3
            elif key == 'BLACKBLOOD':
                value += 2
            elif key == 'PANDORASBOX':
                value += min(3, sum(c in ('Strike_R', 'Defend_R') for c, _ in g.get('deck', ())) / 3)
            row['score'] = value
    return sorted(rows, key=lambda row: (-row['score'], row['action']))


class SurvivalUpperStrategy:
    name = 'overworld-survival'
    version = OVERWORLD_SURVIVAL_VERSION
    profile = SURVIVAL_PROFILE

    def __init__(self, fallback=None, *, trace=None):
        self.fallback, self.trace = fallback, trace
        self._base = OverworldRuleStrategy(fallback, trace=trace)

    def manifest(self):
        return dict(scope='ironclad-public-overworld', acquisition='encounter-coverage-before-engine',
            route='visible-map-coarse-risk-budget', base_events=self._base.version,
            values='uncalibrated-human-priors', private_outcomes=False, profile=asdict(self.profile))

    def decide(self, requests, *, check=lambda: None):
        results, delegated, indices = [None] * len(requests), [], []
        for index, request in enumerate(requests):
            check()
            base = request.env.unwrapped.thick
            state = base.obs_canonical()
            rows = rank_survival(state, np.flatnonzero(request.env.action_masks()),
                                 rmap=base.rmap, current_y=base.gc.cur_map_node_y, profile=self.profile)
            if rows is None:
                delegated.append(request)
                indices.append(index)
                continue
            action = rows[0]['action']
            results[index] = Decision(action, self.name,
                dict(overworld_rule_decisions=1, **{f'overworld_{self.profile.name}_decisions': 1}))
            if self.trace:
                self.trace(dict(kind='overworld_decision', seed=request.seed, step=request.step,
                    phase=state['phase'], version=self.version, ranked=rows, state=state,
                    coverage=asdict(coverage(state)), action=action, source=self.name))
        if delegated:
            decisions = self._base.decide(delegated, check=check)
            for index, decision in zip(indices, decisions):
                results[index] = decision
        return results


class GrowthUpperStrategy(SurvivalUpperStrategy):
    """Same role-aware draft, but spend prepared HP/potions on relics and smiths."""
    name = 'overworld-growth'
    version = OVERWORLD_GROWTH_VERSION
    profile = GROWTH_PROFILE


class ReadinessUpperStrategy(SurvivalUpperStrategy):
    """Survival's draft/rest policy with elite reward gated by public readiness."""
    name = 'overworld-readiness'
    version = OVERWORLD_READINESS_VERSION
    profile = READINESS_PROFILE
