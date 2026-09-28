"""Optional Ironclad engine-building upper policy for A20 experiments.

Every commitment is inferred again from the permanent public deck/relics. There
is no seed, encounter outcome, hidden draw-order, or model access. Scores are
human priors, not estimates of win probability. The existing rules retain map,
event and unsupported-screen behavior; this candidate owns deck construction.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np

from env.core import action_space as A
from env.core.ironclad_cards import IRONCLAD_CARDS
from rl.evaluation.baselines.overworld_rules import (
    CARD_VALUE, CURSES, OverworldRuleStrategy, _item_key,
    cost_only_upgrade_under_snecko, rank_overworld, relic_strength_sources,
    relic_value, remove_value, upgrade_value,
)
from rl.evaluation.strategies import Decision

OVERWORLD_ENGINE_VERSION = 'ironclad-overworld-engine-v2.2-potion-contracts'
OVERWORLD_FUEL_VERSION = 'ironclad-overworld-fuel-v1.1-potion-contracts'
ENERGY_RELICS = frozenset(('CURSEDKEY', 'SOZU', 'COFFEEDRIPPER', 'FUSIONHAMMER',
    'PHILOSOPHERSSTONE', 'RUNICDOME', 'VELVETCHOKER', 'MARKOFPAIN', 'ECTOPLASM', 'BUSTEDCROWN'))
EXHAUST_ONCE = frozenset(('Offering', 'Disarm', 'Shockwave', 'Impervious', 'Fiend Fire',
    'Reaper', 'Feed', 'Pummel', 'Intimidate', 'Seeing Red', 'Warcry', 'Exhume',
    'Infernal Blade', 'Apparition', 'AscendersBane'))
EXHAUST_TOOLS = frozenset(('True Grit', 'Burning Pact', 'Second Wind', 'Fiend Fire'))
DRAW_CARDS = frozenset(('Battle Trance', 'Pommel Strike', 'Shrug It Off', 'Burning Pact',
    'Offering', 'Dark Embrace', 'Brutality', 'Master Of Strategy'))
FRONTLOAD = {'Anger': 1, 'Carnage': 1.6, 'Hemokinesis': 1.4, 'Immolate': 2.3,
    'Fiend Fire': 1.5, 'Pommel Strike': 1, 'Twin Strike': 1, 'Uppercut': 1.2,
    'Cleave': 1, 'Whirlwind': 1.1, 'Bludgeon': 1.4, 'Iron Wave': .6,
    'Perfected Strike': 1, 'Headbutt': .7, 'Wild Strike': 1}
AOE_ATTACKS = frozenset(('Immolate', 'Whirlwind', 'Cleave', 'Combust'))


@dataclass(frozen=True)
class EngineProfile:
    counts: Counter
    relics: frozenset[str]
    act: int
    floor: int
    size: int
    energy: int
    skills: int
    exhaust_fuel: float
    frontload: float
    strength: int
    block_sources: int
    draw: int
    status_sources: int
    commitments: tuple[str, ...]

    def trace(self):
        return dict(commitments=list(self.commitments), deck_size=self.size,
            energy=self.energy, skills=self.skills, exhaust_fuel=self.exhaust_fuel,
            frontload=self.frontload, strength_sources=self.strength,
            block_sources=self.block_sources, draw_sources=self.draw,
            status_sources=self.status_sources)


def engine_profile(state):
    g = state['global']
    cards = g.get('deck', ())
    counts = Counter(c[0] for c in cards)
    relics = frozenset(_item_key(r) for r in g.get('relics', ()))
    skills = sum(n for c, n in counts.items()
        if c in IRONCLAD_CARDS and IRONCLAD_CARDS[c].kind == 'skill')
    fuel = sum(counts[c] * (.35 if c == 'AscendersBane' else 1) for c in EXHAUST_ONCE)
    fuel += sum(counts[c] * 1.6 for c in EXHAUST_TOOLS)
    if counts['Corruption']:
        fuel += min(7, skills * .65)
    strength = sum(counts[c] for c in ('Inflame', 'Spot Weakness', 'Demon Form'))
    strength += relic_strength_sources(state)
    if counts['Rupture'] and any(counts[c] for c in ('Combust', 'Brutality', 'Bloodletting', 'Hemokinesis')):
        strength += 1
    block = sum(n for c, n in counts.items() if c in IRONCLAD_CARDS
        and c != 'Defend_R' and IRONCLAD_CARDS[c].block[0] >= 7)
    status = sum(counts[c] for c in ('Power Through', 'Wild Strike', 'Reckless Charge', 'Immolate'))
    status += int('MARKOFPAIN' in relics)
    tags = []
    if counts['Corruption'] or (fuel >= 3 and (counts['Feel No Pain'] or counts['Dark Embrace'])):
        tags.append('exhaust')
    if strength and any(counts[c] for c in ('Reaper', 'Limit Break', 'Heavy Blade', 'Sword Boomerang', 'Pummel')):
        tags.append('strength')
    if counts['Barricade'] or ('CALIPERS' in relics and block >= 3):
        tags.append('block')
    if counts['Evolve'] and status >= 2:
        tags.append('status')
    return EngineProfile(counts, relics, int(g.get('act', 1)), int(g.get('floor', 1)),
        len(cards), 3 + len(relics & ENERGY_RELICS), skills, fuel,
        sum(n * FRONTLOAD.get(c, 0) for c, n in counts.items()), strength, block,
        sum(counts[c] for c in DRAW_CARDS), status, tuple(tags))


def engine_skip_threshold(profile):
    # A new card must justify delaying the pieces already owned. Exhaust engines
    # tolerate skill fuel; that exception is in card valuation, not a blanket
    # license to bloat the permanent deck with mediocre attacks.
    return 6.2 + .45 * (profile.act - 1) + min(2.2, max(0, profile.size - 18) * .16)


def engine_card_value(card, state, profile=None, *, adding=True):
    p = profile or engine_profile(state)
    c, up = card[:2]
    n, r = p.counts, p.relics
    if c in CURSES:
        return -20.0
    value = float(CARD_VALUE.get(c, 2.5))
    copies = n[c] if adding else max(0, n[c] - 1)
    info = IRONCLAD_CARDS.get(c)
    if info and info.kind == 'attack' and c not in ('Strike_R', 'Bash'):
        if p.act == 1:
            value += max(0, 3.5 - p.frontload) * .65
        elif c not in ('Reaper', 'Feed', 'Fiend Fire', 'Immolate'):
            value -= 1
        if p.frontload >= 4 and c not in ('Reaper', 'Feed', 'Fiend Fire'):
            value -= 1.3
    if c in AOE_ATTACKS and not any(n[x] for x in AOE_ATTACKS):
        value += 1.4 if p.act < 3 else .5
    if c in ('Carnage', 'Hemokinesis', 'Anger', 'Immolate') and p.act == 1 and p.frontload < 2:
        value += 1
    if c == 'Perfected Strike':
        strikes = sum(k for cid, k in n.items() if 'Strike' in cid)
        value += max(-3, (strikes - 6) * .7)
    if c in ('Clash', 'Rampage', 'Searing Blow'):
        value -= 3  # Ascender's Bane and long setup make these speculative.

    # Exhaust is a conversion engine: tools, fuel, block and draw have distinct
    # prerequisites. Ascender's Bane is only one automatic exhaust per combat.
    if c == 'Corruption':
        value = 5 + min(4, max(0, p.skills - 4) * .65)
        value += 2.4 * min(1, n['Feel No Pain']) + 3 * min(1, n['Dark Embrace'])
        value += 7 * int('DEADBRANCH' in r)
    elif c == 'Feel No Pain':
        value = 5.7 + min(5, p.exhaust_fuel * .8)
    elif c == 'Dark Embrace':
        value = 3.8 + min(6, p.exhaust_fuel * .8) + min(1, n['Feel No Pain'])
        value += 1 if p.energy >= 4 else 0
    elif c == 'Burning Pact':
        value = 6.5 + 1.3 * min(1, n['Dark Embrace'] + n['Feel No Pain'])
        value += 1 if p.draw < 2 else 0
    elif c == 'True Grit':
        value = 6 + 1.4 * min(1, n['Dark Embrace'] + n['Feel No Pain'])
    elif c == 'Second Wind':
        value = 6 + 2 * min(1, n['Power Through'])
        value += 1.5 * min(1, n['Feel No Pain']) + 1.3 * min(1, n['Dark Embrace'])
        value += 1.5 * int('RUNICPYRAMID' in r)
    elif c == 'Power Through':
        value = 7.4 + 2 * min(1, n['Second Wind'])
        value += 1.2 * min(1, n['Evolve'] + n['True Grit'] + n['Burning Pact'])
        value -= 1.5 if p.status_sources >= 2 and not n['Evolve'] else 0
    elif c == 'Fiend Fire':
        value += 1.5 + min(2, p.draw * .6) + min(2, p.strength)
    elif c == 'Exhume':
        value = 3 + 2.7 * min(2, sum(n[x] for x in ('Offering', 'Impervious', 'Reaper', 'Feed', 'Fiend Fire')))
    elif c == 'Sentinel':
        value = 3 + 4 * int(bool(n['Corruption'] or n['Second Wind'] or n['Burning Pact']))
    elif c in ('Intimidate', 'Warcry', 'Seeing Red', 'Infernal Blade'):
        value += min(3.6, 1.8 * n['Dark Embrace'] + 1.4 * n['Feel No Pain'])
    if info and info.kind == 'skill' and n['Corruption'] and c not in ('Flex', 'Clash'):
        value += 1.3

    # Strength's payoff is repeated hits and recovery; Limit Break without a
    # persistent source remains a dead card. A second expensive setup power does
    # not solve a missing attack or block turn.
    if c in ('Inflame', 'Spot Weakness'):
        value += 2 if not p.strength and p.act > 1 else .3
        value += 1.5 * min(1, n['Reaper'])
    elif c == 'Demon Form':
        value = 7.2 + 2 * min(1, n['Reaper']) - 2.5 * bool(p.strength)
        value -= 1.5 if p.energy == 3 and p.block_sources < 2 else 0
    elif c == 'Limit Break':
        value = 1 if not p.strength else 7.2 + min(2.5, p.strength * .8)
        value += 1.3 * min(1, n['Reaper']) if p.strength else 0
    elif c in ('Heavy Blade', 'Sword Boomerang', 'Pummel'):
        value += min(5, p.strength * 2.2)
    elif c == 'Reaper':
        value = 6.5 + min(7, p.strength * 2.5)
        value += 1 if state['global']['hp'] < .65 * state['global']['max_hp'] else 0
    elif c == 'Rupture':
        repeat = sum(n[x] for x in ('Combust', 'Brutality', 'Bloodletting'))
        value = 2 + min(7, repeat * 3 + n['Hemokinesis'])

    # Block/status payoffs are acquired after their generators. Corruption and
    # Barricade may coexist; speculative Entrench alone is never an engine.
    if c == 'Barricade':
        value = 2.5 + min(3, p.block_sources * .7)
        value += 3 * int(bool(n['Corruption'] and n['Feel No Pain']))
        value += 1.5 * min(1, n['Impervious'])
    elif c == 'Entrench':
        value = 2 + 5 * int('block' in p.commitments) + min(2, p.block_sources * .4)
    elif c == 'Body Slam':
        value = 3 + min(3, p.block_sources * .8) + 2.8 * int('block' in p.commitments)
        value += 1.5 * int(bool(n['Second Wind'] and n['Power Through']))
    elif c == 'Juggernaut':
        value = 2.5 + 4 * int(bool(n['Feel No Pain'] and p.exhaust_fuel >= 4))
        value += min(1.5, n['Metallicize'] * .7)
    elif c == 'Evolve':
        value = 4 + min(5, p.status_sources * 2)
    elif c == 'Fire Breathing':
        value = 2 + min(5, p.status_sources * 1.5) + 1.5 * min(1, n['Evolve'])
    elif c == 'Brutality':
        value = 5.5 + 1.5 * min(1, n['Rupture']) + int(p.energy >= 4)
    elif c == 'Bloodletting':
        value = 4.5 + min(3, p.draw * .8) + int('RUNICPYRAMID' in r)
    elif c == 'Battle Trance':
        value = 8.5 - 2 * int(bool(n['Dark Embrace'] and p.exhaust_fuel >= 5))
    elif c == 'Feed':
        value = 10 - max(0, p.floor - 12) * .19

    if c in ('Offering', 'Shrug It Off', 'Pommel Strike'):
        value += .7
    if c in DRAW_CARDS and p.draw < 2 and p.size >= 15:
        value += 1
    if info and info.kind == 'power' and p.energy == 3 and not up:
        expensive_setup = sum(n[x] for x in ('Corruption', 'Demon Form', 'Barricade', 'Dark Embrace'))
        if info.cost[0] >= 2 and expensive_setup >= 2:
            value -= 2
    if 'SNECKOEYE' in r and info:
        value += 1.8 if info.cost[0] >= 2 else -1.4 if info.cost[0] == 0 else 0
    # Only repeatable scaling/payoff cards saturate sharply. Multiple cheap draw
    # and block cards remain useful when a large exhaust engine consumes fuel.
    duplicate_penalty = 2.7 if c in ('Corruption', 'Dark Embrace', 'Demon Form', 'Barricade',
        'Evolve', 'Fire Breathing', 'Limit Break', 'Anger', 'Battle Trance', 'Armaments') else 1.3
    if c in ('Shrug It Off', 'Burning Pact', 'Offering', 'Disarm'):
        duplicate_penalty = .65
    value -= duplicate_penalty * copies
    value += boss_card_adjustment(card, state, p)
    if up:
        value += min(1.8, engine_upgrade_value((c, 0), state, p) / 6)
    return float(value)


def _public_status_boss(state):
    g = state['global']
    boss = _item_key(g.get('boss', ''))
    return (boss in ('HEXAGHOST', 'SLIMEBOSS', 'DONUANDDECA') or
        (boss == 'TIMEEATER' and int(g.get('ascension', 0)) >= 19))


def boss_card_adjustment(card, state, profile=None):
    """Deck-readiness-gated priors for the currently displayed public boss.

    Boss-created statuses are not permanent-deck generators. They justify one
    answer for the upcoming fight, not an unlimited status-engine commitment.
    At A20 this reads only the displayed boss, never the unrevealed second one.
    """
    p = profile or engine_profile(state)
    c = card[0]
    n, g = p.counts, state['global']
    boss = _item_key(g.get('boss', ''))
    if p.act == 1 and (p.floor < 8 or p.frontload < 2):
        return 0.0
    late = p.floor >= {1: 10, 2: 27, 3: 43}.get(p.act, 43)
    scale = 1.0 if late else .7
    statuses = _public_status_boss(state)
    bonus = 0.0
    if statuses and not n[c]:
        if c == 'Evolve':
            bonus += 3.5 * scale
        elif c == 'Fire Breathing' and boss in ('HEXAGHOST', 'SLIMEBOSS'):
            bonus += 4.8 * scale
        elif c == 'True Grit' and card[1] and not any(n[x] for x in EXHAUST_TOOLS):
            bonus += 1.3 * scale
    if boss in ('HEXAGHOST', 'CHAMP') and not p.strength:
        if c in ('Inflame', 'Spot Weakness'):
            bonus += (2.8 if boss == 'CHAMP' else 2.2) * scale
        elif c == 'Demon Form' and p.act > 1 and (p.energy >= 4 or p.block_sources >= 3):
            bonus += 2.8 * scale
    if boss == 'CHAMP' and p.strength and c == 'Limit Break' and not n[c]:
        bonus += 1.5 * scale
    if p.act == 1 and p.block_sources < 2 and not n[c] and c in ('True Grit', 'Ghostly Armor'):
        # A compact deck still needs playable defense. Do not wait for a rare
        # engine payoff while passing the first useful bridge block cards.
        bonus = max(bonus, 1.5 * scale)
    if boss == 'THEGUARDIAN' and p.block_sources < 2 and c in (
            'Shrug It Off', 'Flame Barrier', 'Ghostly Armor', 'Power Through', 'Impervious'):
        bonus = max(bonus, 1.8 * scale)
    if boss in ('HEXAGHOST', 'THEGUARDIAN', 'AWAKENEDONE') and c == 'Disarm' and not n[c]:
        bonus += 1.5 * scale
    if boss == 'AWAKENEDONE' and c in ('Metallicize', 'Brutality', 'Combust', 'Fire Breathing'):
        # Do not penalize Corruption / Dark Embrace / Feel No Pain or the first
        # persistent strength source that makes the deck work across both lives.
        bonus -= 1.8 * scale
    return bonus


def engine_upgrade_value(card, state, profile=None):
    p = profile or engine_profile(state)
    c, up = card[:2]
    if up and c != 'Searing Blow':
        return 0.0
    if cost_only_upgrade_under_snecko(card, state):
        return 0.0
    value = upgrade_value(card, state)
    if c in ('Corruption', 'Dark Embrace'):
        value += 4 if p.energy == 3 else 1
    if c == 'True Grit':
        value += 2  # Targeted exhaust protects the actual engine pieces.
    if c == 'Burning Pact':
        value += 4
    if c == 'Limit Break' and p.strength:
        value += 4
    if c == 'Body Slam' and ('block' in p.commitments or p.block_sources >= 3):
        value += 5
    if c == 'Evolve':
        boss_ready = p.act > 1 or (p.floor >= 8 and p.frontload >= 2)
        if p.status_sources or (boss_ready and _public_status_boss(state)):
            value += 4
    return float(value)


def engine_remove_value(card, state, profile=None, *, card_value_fn=engine_card_value):
    p = profile or engine_profile(state)
    c, up = card[:2]
    if c in CURSES or c == 'Strike_R':
        return remove_value(card, state) + (1.5 if c == 'Strike_R' and p.frontload >= 3 else 0)
    if c == 'Defend_R':
        return (3 if p.counts['Corruption'] and p.skills < 9 else 6) - 2 * bool(up)
    return 6 - card_value_fn(card, state, p, adding=False)


def engine_relic_value(relic, state, profile=None):
    p = profile or engine_profile(state)
    r, n = _item_key(relic), p.counts
    value = relic_value(relic, state)
    if r == 'DEADBRANCH':
        value += min(5, p.exhaust_fuel) + 5 * bool(n['Corruption'])
    elif r == 'RUNICPYRAMID':
        value += 3 if p.energy >= 4 or n['Corruption'] or n['Second Wind'] else 0
    elif r == 'VELVETCHOKER':
        value -= min(8, p.draw + 2 * n['Dark Embrace'] + 3 * n['Corruption'])
    elif r == 'SNECKOEYE':
        value -= 3 if 'exhaust' in p.commitments and p.energy >= 4 else 0
    elif r == 'MARKOFPAIN':
        value += 5 if n['Evolve'] or ('MEDICALKIT' in p.relics and n['Dark Embrace']) else 0
    elif r == 'COFFEEDRIPPER':
        value += 2 if p.strength and n['Reaper'] else 0
    elif r == 'EMPTYCAGE':
        value += 2 if p.size <= 20 and p.commitments else 0
    elif r == 'MEDICALKIT':
        value += 3 * bool(n['Dark Embrace']) + 2 * bool(n['Feel No Pain'])
    elif r == 'RUNICDOME':
        # Current combat planner requires intents; preserve the shared fallback
        # penalty instead of pretending an energy gain is freely executable.
        return value
    if r in ENERGY_RELICS and p.energy == 3:
        value += 1.5 if p.draw >= 2 or any(n[c] for c in ('Corruption', 'Demon Form', 'Barricade')) else .5
    return float(value)


def fuel_card_value(card, state, profile=None, *, adding=True):
    """Value supported conversion fuel, without rewarding arbitrary deck growth.

    Corruption makes skills free; Feel No Pain / Dark Embrace convert their
    exhaustion into block / draw. A large deck still needs access to that setup,
    so extra fuel requires draw support as well as an actual conversion engine.
    """
    p = profile or engine_profile(state)
    value = engine_card_value(card, state, p, adding=adding)
    c, up = card[:2]
    n, relics = p.counts, p.relics
    opening_ready = p.act > 1 or p.frontload >= 2
    if not opening_ready:
        return value
    copies = n[c] if adding else max(0, n[c] - 1)
    supported = bool(n['Corruption'] and (n['Feel No Pain'] or n['Dark Embrace']))
    draw_ready = bool(n['Dark Embrace'] or p.draw >= 3 or 'SNECKOEYE' in relics)
    info = IRONCLAD_CARDS.get(c)
    useful_skill = info is not None and info.kind == 'skill' and (
        info.block[bool(up)] > 0 or c in ('Burning Pact', 'Battle Trance', 'Offering',
            'Disarm', 'Shockwave', 'Intimidate', 'Seeing Red') or (c == 'Warcry' and up))
    if supported and draw_ready and useful_skill:
        # Cancel only this fuel card's raw-size surcharge. Ordinary attacks,
        # unknown skills and speculative payoffs still pay the full dilution cost.
        value += min(2.2, max(0, p.size - 18) * .16)
        value += .8 * min(2, n['Feel No Pain']) + 1.0 * min(1, n['Dark Embrace'])
        value += .8 * int('CHARONSASHES' in relics)
        # Duplicate free skills remain fuel, but cannot grow without bound.
        if copies >= 3:
            value -= 1.5 * (copies - 2)
    if c == 'Feel No Pain' and supported and draw_ready and p.skills >= 8 and 1 <= copies < 3:
        # Stacks add real block per exhaust; they are not redundant on/off powers.
        value += min(3.2, p.skills * .25)
    if info is not None and info.kind == 'power' and value >= 6:
        # These bonuses need a worthwhile power, not merely the POWER card type.
        # Canonical "Bird Faced Urn" / "Mummified Hand" and native enum spellings
        # share the existing punctuation-insensitive relic key normalizer.
        value += 1.6 * int('MUMMIFIEDHAND' in relics)
        value += 1.2 * int('BIRDFACEDURN' in relics)
    return float(value)


def fuel_relic_value(relic, state, profile=None):
    p = profile or engine_profile(state)
    value = engine_relic_value(relic, state, p)
    powers = sum(n for c, n in p.counts.items()
        if c in IRONCLAD_CARDS and IRONCLAD_CARDS[c].kind == 'power')
    key = _item_key(relic)
    if key == 'MUMMIFIEDHAND':
        value += min(5, powers * .9)
    elif key == 'BIRDFACEDURN':
        value += min(5, powers)
    elif key == 'CHARONSASHES':
        value += min(4, p.exhaust_fuel * .4)
    return float(value)


def rank_engine_overworld(state, legal, *, rmap=None, current_y=-1,
        card_value_fn=engine_card_value, relic_value_fn=engine_relic_value):
    ranked = rank_overworld(state, legal, rmap=rmap, current_y=current_y)
    if ranked is None:
        return None
    original = ranked[0]['action']
    p, opts = engine_profile(state), state.get('options', {})
    context = opts.get('select', {}).get('context')
    threshold = engine_skip_threshold(p)
    touched = False
    for row in ranked:
        kind, index = A.segment_of(row['action'])
        old = row['score']
        if kind == 'REWARD_CARD':
            row['score'] = card_value_fn(opts['cards'][index], state, p) - threshold
        elif kind == 'SHOP_CARD':
            cid, up, price = opts['shop_cards'][index]
            row['score'] = (card_value_fn((cid, up), state, p) - threshold) * 2 - price / 35
        elif kind == 'BOSS_RELIC':
            row['score'] = relic_value_fn(state['global']['boss_relics'][index], state, p)
        elif kind == 'SHOP_RELIC':
            relic, price = opts['shop_relics'][index]
            row['score'] = relic_value_fn(relic, state, p) - price / 35
        elif kind == 'SHOP_REMOVE':
            row['score'] = max((engine_remove_value(c, state, p, card_value_fn=card_value_fn) for c in state['global'].get('deck', ())), default=0) - opts['shop_remove_cost'] / 20
        elif kind == 'PICK_CARD':
            card = opts['cards'][index]
            if context == 'UPGRADE':
                row['score'] = engine_upgrade_value(card, state, p)
            elif context in ('REMOVE', 'TRANSFORM', 'TRANSFORM_UPGRADE'):
                row['score'] = engine_remove_value(card, state, p, card_value_fn=card_value_fn)
            elif context in ('OBTAIN', 'DUPLICATE'):
                row['score'] = card_value_fn(card, state, p)
            else:
                continue
        elif kind == 'CAMPFIRE' and A._CAMP_NAME_BY_IDX[index] in ('smith', 'toke'):
            values = (engine_upgrade_value(c, state, p) if A._CAMP_NAME_BY_IDX[index] == 'smith'
                else engine_remove_value(c, state, p, card_value_fn=card_value_fn)
                for c in state['global'].get('deck', ()))
            row['score'] = max(values, default=0)
        else:
            continue
        row['v1_score'] = old
        touched = True
    ranked.sort(key=lambda row: (-row['score'], row['action']))
    return ranked, original, p, touched


class EngineUpperStrategy:
    name = 'overworld-engine'
    version = OVERWORLD_ENGINE_VERSION
    draft_profile = 'compact'
    card_value_fn = staticmethod(engine_card_value)
    relic_value_fn = staticmethod(engine_relic_value)

    def __init__(self, fallback=None, *, trace=None):
        self.fallback, self.trace = fallback, trace
        self._base = OverworldRuleStrategy(fallback, trace=trace)

    def manifest(self):
        return dict(scope='deck-acquisition-removal-upgrade-and-relics',
            base_version=self._base.version, draft_profile=self.draft_profile,
            commitment='public-deck-derived',
            card_threshold='act-and-deck-size', boss_prior='displayed-boss-after-opening-readiness',
            future_information=False)

    def decide(self, requests, *, check=lambda: None):
        result, delegated, indices = [None] * len(requests), [], []
        for index, request in enumerate(requests):
            check()
            base = request.env.unwrapped.thick
            state = base.obs_canonical()
            ranking = rank_engine_overworld(state, np.flatnonzero(request.env.action_masks()),
                rmap=base.rmap, current_y=base.gc.cur_map_node_y,
                card_value_fn=self.card_value_fn, relic_value_fn=self.relic_value_fn)
            if ranking is None:
                delegated.append(request)
                indices.append(index)
                continue
            rows, original, profile, touched = ranking
            action = rows[0]['action']
            diagnostics = dict(overworld_rule_decisions=1, overworld_engine_decisions=int(touched),
                overworld_engine_changed_decisions=int(action != original))
            if self.draft_profile == 'fuel':
                diagnostics['overworld_fuel_decisions'] = int(touched)
            for family in profile.commitments:
                diagnostics[f'overworld_engine_{family}_decisions'] = int(touched)
            result[index] = Decision(action, self.name, diagnostics)
            if self.trace:
                self.trace(dict(kind='overworld_decision', seed=request.seed, step=request.step,
                    phase=state['phase'], version=self.version, ranked=rows, state=state,
                    action=action, source=self.name, v1_action=original,
                    engine=profile.trace(), draft_profile=self.draft_profile,
                    public_boss=state['global'].get('boss'),
                    skip_threshold=engine_skip_threshold(profile)))
        if delegated:
            decisions = self._base.decide(delegated, check=check)
            if len(decisions) != len(delegated):
                raise ValueError('engine upper fallback returned a wrong decision count')
            for index, decision in zip(indices, decisions):
                result[index] = decision
        return result


class FuelUpperStrategy(EngineUpperStrategy):
    """Independent exhaust-fuel challenger; compact engine remains the control."""
    name = 'overworld-fuel'
    version = OVERWORLD_FUEL_VERSION
    draft_profile = 'fuel'
    card_value_fn = staticmethod(fuel_card_value)
    relic_value_fn = staticmethod(fuel_relic_value)
