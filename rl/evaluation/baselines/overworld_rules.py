"""Public-state Ironclad deck building and route heuristics.

The scores express preferences, not learned values. Unknown event semantics are
delegated explicitly; the simulator is never stepped, forked, or queried for
unrevealed outcomes. Map topology is public in the game.
"""
from __future__ import annotations

from collections import Counter
from functools import lru_cache

import numpy as np

from env.core import action_space as A
from env.core.ironclad_cards import IRONCLAD_CARDS
from env.core.spiremap import MAX_Y

OVERWORLD_RULE_VERSION = 'ironclad-overworld-v1.2-potion-contracts'
CURSES = frozenset(('Injury', 'Doubt', 'Shame', 'Pain', 'Regret', 'Decay',
    'Normality', 'Clumsy', 'CurseOfTheBell', 'Parasite', 'Writhe', 'Pride',
    'Necronomicurse', 'AscendersBane'))
PERMANENT_CURSES = frozenset(('CurseOfTheBell', 'Necronomicurse', 'AscendersBane'))
AOE = frozenset(('Cleave', 'Whirlwind', 'Immolate', 'Combust', 'Reaper'))
SCALING = frozenset(('Inflame', 'Spot Weakness', 'Demon Form', 'Limit Break',
    'Barricade', 'Corruption', 'Feel No Pain', 'Dark Embrace'))
STRENGTH = frozenset(('Inflame', 'Spot Weakness', 'Demon Form'))
EXHAUST = frozenset(('True Grit', 'Burning Pact', 'Fiend Fire', 'Second Wind',
    'Corruption', 'Offering', 'Disarm', 'Impervious', 'Shockwave'))
DRAW = frozenset(('Pommel Strike', 'Shrug It Off', 'Battle Trance', 'Burning Pact',
    'Offering', 'Dark Embrace', 'Master Of Strategy'))

# Standalone value before deck needs, duplicate penalties, and upgrades.
CARD_VALUE = {
    'Strike_R': 1, 'Defend_R': 2, 'Bash': 4.5, 'Anger': 6, 'Armaments': 5,
    'Body Slam': 3, 'Clash': -1, 'Cleave': 5.8, 'Clothesline': 5.2, 'Flex': 3,
    'Havoc': 2, 'Headbutt': 5, 'Heavy Blade': 3.5, 'Iron Wave': 5,
    'Perfected Strike': 4.5, 'Pommel Strike': 6.6, 'Shrug It Off': 7,
    'Sword Boomerang': 4.5, 'Thunderclap': 3.5, 'True Grit': 5.6,
    'Twin Strike': 5.3, 'Warcry': 1, 'Wild Strike': 3, 'Battle Trance': 8,
    'Blood For Blood': 3, 'Bloodletting': 4, 'Burning Pact': 5.5, 'Carnage': 7,
    'Combust': 5.5, 'Dark Embrace': 4, 'Disarm': 8, 'Dropkick': 4,
    'Dual Wield': 2.5, 'Entrench': 0, 'Evolve': 3.5, 'Feel No Pain': 5,
    'Fire Breathing': 3.5, 'Flame Barrier': 7, 'Ghostly Armor': 5.5,
    'Hemokinesis': 6.5, 'Infernal Blade': 3, 'Inflame': 7.5, 'Intimidate': 3,
    'Metallicize': 5, 'Power Through': 6.8, 'Pummel': 4.5, 'Rage': 3,
    'Rampage': 3, 'Reckless Charge': 3, 'Rupture': 1, 'Searing Blow': 1,
    'Second Wind': 5, 'Seeing Red': 3.5, 'Sentinel': 3, 'Sever Soul': 3,
    'Shockwave': 8.5, 'Spot Weakness': 7.5, 'Uppercut': 7, 'Whirlwind': 7,
    'Barricade': 3, 'Berserk': 1, 'Bludgeon': 6, 'Brutality': 4,
    'Corruption': 6, 'Demon Form': 6.5, 'Double Tap': 4, 'Exhume': 3,
    'Feed': 8, 'Fiend Fire': 8, 'Immolate': 9, 'Impervious': 9,
    'Juggernaut': 3, 'Limit Break': 1, 'Offering': 9, 'Reaper': 7,
    'Apotheosis': 10, 'Master Of Strategy': 8, 'Bandage Up': 6,
    'Flash Of Steel': 6, 'Finesse': 6, 'Hand Of Greed': 7,
    'Trip': 5, 'Blind': 5, 'Panacea': 4, 'Good Instincts': 4,
    'Bite': 5, 'Apparition': 8, 'Ritual Dagger': 5,
}


def deck_counts(state):
    return Counter(card[0] for card in state['global'].get('deck', ()))


def _item_key(name):
    # CommunicationMod ids vary in spaces/apostrophes; native enum names use
    # underscores. Egg ids additionally retain the historical suffix " 2".
    return ''.join(c for c in name.upper() if c.isalnum()).removesuffix('2')


def has_relic(state, *names):
    held = {_item_key(name) for name in state['global'].get('relics', ())}
    return bool(held & {_item_key(name) for name in names})


EMPTY_POTION_SLOTS = frozenset(('', 'EMPTY_POTION_SLOT'))


def can_obtain_potion(state):
    """A public empty usable slot and no Sozu are both required."""
    return (not has_relic(state, 'SOZU') and
        any(p in EMPTY_POTION_SLOTS for p in state['global'].get('potions', ())))


COST_ONLY_UPGRADES = frozenset(('Body Slam', 'Corruption', 'Dark Embrace',
    'Barricade', 'Entrench', 'Infernal Blade', 'Seeing Red', 'Exhume', 'Apotheosis'))


def cost_only_upgrade_under_snecko(card, state):
    """Confused overwrites printed energy cost when the card is drawn."""
    return card[0] in COST_ONLY_UPGRADES and has_relic(state, 'SNECKO_EYE')


def relic_strength_sources(state):
    """Count public, persistent relic sources; this is not a Strength amount.

    Girya requires an actual lift counter. Du-Vu Doll requires an actual curse;
    carrying the relic alone is insufficient. Brimstone supplies strength every
    turn. Red Skull / Sling of Courage are conditional and excluded here.
    """
    g = state['global']
    relics = tuple(_item_key(r) for r in g.get('relics', ()))
    counters = dict(zip(relics, g.get('relic_counters', ())))
    held = set(relics)
    return (int('VAJRA' in held) + int('BRIMSTONE' in held) +
        int('GIRYA' in held and counters.get('GIRYA', 0) > 0) +
        int('DUVUDOLL' in held and any(c[0] in CURSES for c in g.get('deck', ()))))


def card_value(card, state, *, adding=True):
    """Value of a displayed card, including fit and redundancy in this deck."""
    cid, upgraded = card[:2]
    counts = deck_counts(state)
    g = state['global']
    act = int(g.get('act', 1))
    if cid in CURSES:
        return -20
    value = CARD_VALUE.get(cid, 2.5)
    count = counts[cid] if adding else max(0, counts[cid] - 1)
    value -= count * (2.5 if cid in SCALING or cid in ('Anger', 'Armaments', 'Battle Trance') else 1.2)
    attacks = sum(n for c, n in counts.items() if c not in ('Strike_R', 'Bash')
                  and c in IRONCLAD_CARDS and IRONCLAD_CARDS[c].kind == 'attack')
    if cid in IRONCLAD_CARDS:
        info = IRONCLAD_CARDS[cid]
        if info.kind == 'attack' and cid not in ('Strike_R', 'Bash'):
            if act == 1:
                value += max(0, 3 - attacks) * .65
            else:
                value -= .8 if cid not in AOE and cid not in ('Reaper', 'Feed') else 0
        if info.block[1] > 7:
            defense = sum(n for c, n in counts.items() if c in IRONCLAD_CARDS
                          and IRONCLAD_CARDS[c].block[1] > 7 and c != 'Defend_R')
            value += max(0, act + 1 - defense) * .45
        value += min(1.5, upgrade_value((cid, 0), state) / 7) if upgraded else 0
    if cid in AOE and not any(counts[c] for c in AOE):
        value += 1.8
    strength = sum(counts[c] for c in STRENGTH) + relic_strength_sources(state)
    if cid in STRENGTH and not strength:
        value += 2 if act > 1 else .5
    if cid in ('Heavy Blade', 'Sword Boomerang', 'Pummel', 'Reaper', 'Limit Break'):
        value += min(5, strength * (4 if cid == 'Limit Break' else 1.8))
    if cid == 'Limit Break' and not strength:
        value -= 6
    exhaust = sum(counts[c] for c in EXHAUST)
    if cid in ('Feel No Pain', 'Dark Embrace'):
        value += min(4, exhaust * .7)
    if cid in EXHAUST:
        value += min(2, counts['Feel No Pain'] + counts['Dark Embrace'])
    if cid == 'Corruption':
        value += 3 if has_relic(state, 'DEAD_BRANCH') else 0
    if cid in ('Body Slam', 'Entrench', 'Barricade'):
        value += min(4, 2 * counts['Barricade'] + counts['Impervious'] + counts['Power Through'])
    if cid == 'Perfected Strike':
        value += max(-3, (sum(n for c, n in counts.items() if 'Strike' in c) - 6) * .6)
    if cid == 'Rupture':
        value += min(5, 2 * sum(counts[c] for c in ('Combust', 'Brutality', 'Hemokinesis', 'Offering')))
    if cid in ('Evolve', 'Fire Breathing'):
        value += min(3, sum(counts[c] for c in ('Power Through', 'Wild Strike', 'Reckless Charge')))
    if cid == 'Demon Form' and strength:
        value -= 3
    if cid == 'Feed':
        value -= max(0, int(g.get('floor', 1)) - 15) * .12
    if adding and len(g.get('deck', ())) > 23 and cid not in DRAW:
        value -= min(3, (len(g['deck']) - 23) * .2)
    return value


def upgrade_value(card, state):
    cid, upgraded = card[:2]
    if upgraded and cid != 'Searing Blow':
        return 0.0
    if cost_only_upgrade_under_snecko(card, state):
        return 0.0
    special = {'Bash': 7, 'Armaments': 8, 'True Grit': 8, 'Pommel Strike': 8,
        'Uppercut': 10, 'Shockwave': 9, 'Offering': 9, 'Whirlwind': 9,
        'Apotheosis': 12, 'Body Slam': 6, 'Corruption': 9, 'Dark Embrace': 9,
        'Battle Trance': 5, 'Limit Break': 6, 'Reaper': 2, 'Feed': 6}
    if cid in special:
        return float(special[cid])
    info = IRONCLAD_CARDS.get(cid)
    if not info:
        return 1.0
    return (info.damage[1] - info.damage[0]) * .65 + (info.block[1] - info.block[0]) * .85 + max(0, info.cost[0] - info.cost[1]) * (0 if has_relic(state, 'SNECKO_EYE') else 6) + (2 if info.kind == 'power' else 0)


def remove_value(card, state):
    cid, upgraded = card[:2]
    if cid in PERMANENT_CURSES:
        return -100
    if cid in CURSES:
        return 25 + (5 if cid in ('Normality', 'Pain', 'Regret') else 0)
    if cid == 'Strike_R':
        return 9 - 2 * bool(upgraded) - 2 * bool(deck_counts(state)['Perfected Strike'])
    if cid == 'Defend_R':
        return 6 - 2 * bool(upgraded)
    return 5 - card_value(card, state, adding=False)


def relic_value(relic, state):
    g = state['global']
    counts = deck_counts(state)
    values = {'SNECKO_EYE': 10, 'RUNIC_PYRAMID': 8, 'CURSED_KEY': 11,
        'SOZU': 7, 'COFFEE_DRIPPER': 3, 'FUSION_HAMMER': 8,
        'PHILOSOPHERS_STONE': 5, 'RUNIC_DOME': -8, 'VELVET_CHOKER': 8,
        'MARK_OF_PAIN': 4, 'ECTOPLASM': 5, 'SLAVERS_COLLAR': 9,
        'BUSTED_CROWN': 1, 'BLACK_BLOOD': 9, 'BLACK_STAR': 3,
        'PANDORAS_BOX': 9, 'ASTROLABE': 8, 'EMPTY_CAGE': 5,
        'CALLING_BELL': 5, 'SACRED_BARK': 5, 'TINY_HOUSE': 4,
        'ORRERY': 5, 'DEAD_BRANCH': 10, 'ANCHOR': 7, 'BAG_OF_PREPARATION': 8,
        'BAG_OF_MARBLES': 7, 'VAJRA': 7, 'ODDLY_SMOOTH_STONE': 7,
        'ICE_CREAM': 8, 'TOXIC_EGG': 9, 'MOLTEN_EGG': 7, 'FROZEN_EGG': 5,
        'CHEMICAL_X': 4, 'MEDICAL_KIT': 5, 'ORICHALCUM': 5,
        'PRESERVED_INSECT': 5, 'MANGO': 6, 'PEAR': 4, 'STRAWBERRY': 3,
        'MEAT_ON_THE_BONE': 10, 'TOY_ORNITHOPTER': 8, 'MAGIC_FLOWER': 9,
        'CALIPERS': 3, 'MEMBERSHIP_CARD': 10, 'THE_COURIER': 5}
    relic = {_item_key(name): name for name in values}.get(_item_key(relic), relic)
    value = float(values.get(relic, 4))
    if relic == 'SNECKO_EYE':
        known = [IRONCLAD_CARDS[c].cost[bool(u)] for c, u in g.get('deck', ()) if c in IRONCLAD_CARDS]
        value += 8 * ((sum(known) / max(1, len(known))) - 1.3)
        value -= 8 if has_relic(state, 'RUNIC_PYRAMID') else 0
    if relic == 'RUNIC_PYRAMID':
        value -= 6 if has_relic(state, 'SNECKO_EYE') else 0
        value += min(2, counts['Second Wind'] + counts['Fiend Fire'] + counts['True Grit'])
    if relic == 'COFFEE_DRIPPER':
        value += 4 if counts['Reaper'] or has_relic(state, 'MEAT_ON_THE_BONE', 'ETERNAL_FEATHER', 'BLACK_BLOOD') else 0
    if relic == 'VELVET_CHOKER':
        value -= min(7, counts['Anger'] * 2 + counts['Corruption'] * 4 + counts['Offering'] * 2)
    if relic == 'BUSTED_CROWN':
        value += 5 if len(g.get('deck', ())) >= 22 and any(counts[c] for c in SCALING) else 0
    if relic == 'CHEMICAL_X':
        value += 5 * min(1, counts['Whirlwind'])
    if relic == 'CALIPERS':
        value += 4 * min(1, counts['Impervious'])
    return value


def route_scores(state, choices, rmap, current_y):
    """Discounted public-map utility, recalculated after each actual room."""
    g = state['global']
    hp = g['hp'] / max(1, g['max_hp'])
    act = g.get('act', 1)
    upgrades = sorted((upgrade_value(c, state) for c in g.get('deck', ())), reverse=True)
    strength = sum(card_value(c, state, adding=False) >= 7 for c in g.get('deck', ()))
    rest_heal = 0 if has_relic(state, 'COFFEE_DRIPPER', 'MARK_OF_THE_BLOOM') else 8 * max(0, .9 - hp)
    smith = 0 if has_relic(state, 'FUSION_HAMMER') else min(2, sum(upgrades[:2]) / 8)
    utility = {'REST': rest_heal + smith,
        'ELITE': (-7 if hp < .55 else -3 if hp < .8 else 1) + min(2, strength / 4) - (act - 1),
        'MONSTER': 1.7 if act == 1 and len(g.get('deck', ())) < 18 else -.5,
        'EVENT': 1.2, 'SHOP': min(5, max(-3, (g.get('gold', 0) - 80) / 55)),
        'TREASURE': 3}

    @lru_cache(None)
    def score(x, y):
        if y > MAX_Y:
            return 0.0
        own = utility.get(rmap.room_type(x, y), 0)
        children = rmap.reachable(x, y) if y < MAX_Y else []
        return own + (.8 * max(score(child, y + 1) for child in children) if children else 0)

    return {action: score(column, max(0, current_y + 1)) for column, action in choices}


def event_scores(state):
    """Verified button meanings in GameContext::chooseEventOption.

    Multi-stage unknown events use the model; no event-data union fields read.
    """
    g, event = state['global'], state['options'].get('event', '')
    hp = g['hp'] / max(1, g['max_hp'])
    curses = any(c in CURSES - PERMANENT_CURSES for c, _ in g.get('deck', ()))
    fixed = {'OMINOUS_FORGE': {0: 8, 1: -6, 2: 0},
        'BIG_FISH': {0: 10 if hp < .72 else 0, 1: 5, 2: -3},
        'GOLDEN_SHRINE': {0: 8, 1: 1, 2: 0}, 'PURIFIER': {0: 10, 1: 0},
        'TRANSMORGRIFIER': {0: 8, 1: 0}, 'UPGRADE_SHRINE': {0: 8, 1: 0},
        'LIVING_WALL': {0: 12 if curses else 6, 1: 5, 2: 8},
        'THE_DIVINE_FOUNTAIN': {0: 10, 1: 0}, 'THE_SSSSSERPENT': {0: -2, 1: 0},
        'OLD_BEGGAR': {0: 8 if curses else 1, 1: 0},
        'THE_CLERIC': {0: 12 if hp < .65 else 1, 1: 10 if curses else 5, 2: 0},
        'THE_LIBRARY': {0: 7, 1: 12 if hp < .55 else 0},
        'THE_NEST': {0: 5, 1: 1}, 'KNOWING_SKULL': {3: 10},
        'THE_MAUSOLEUM': {0: -5, 1: 0}, 'DEAD_ADVENTURER': {0: -1, 1: 0},
        'PLEADING_VAGRANT': {0: 8, 1: -4, 2: 0},
        'SHINING_LIGHT': {0: 5 if hp > .65 else -5, 1: 0},
        'WORLD_OF_GOOP': {0: 3 if g['hp'] > 35 else -5, 1: 0},
        'GOLDEN_IDOL': {0: 8, 1: 0, 2: -10, 3: 3 if hp > .45 else -8, 4: 0},
        'WINDING_HALLS': {0: 1 if hp > .55 else -5, 1: -2, 2: 0},
        'WING_STATUE': {0: 5 if g['hp'] > 20 else -5, 1: 6, 2: 0},
        'THE_MOAI_HEAD': {0: 6 if hp < .45 else -5, 1: 8, 2: 0},
        'GHOSTS': {0: -1, 1: 0}, 'SCRAP_OOZE': {0: 1 if hp > .8 else -2, 1: 0},
        'ANCIENT_WRITING': {0: 12 if curses else 2, 1: 6},
        'THE_JOUST': {0: 1, 1: 2}, 'WHEEL_OF_CHANGE': {0: 1},
        'MINDBLOOM': {0: 8 if hp > .45 else 2, 1: -10, 2: -8, 3: 6 if hp < .5 else -3},
        'CURSED_TOME': {0: 6 if g['hp'] > 35 else -5, 1: 0,
            2: 3, 3: 3, 4: 3, 5: 8 if g['hp'] > 20 else -5, 6: 0}}
    # The canonical API uses eventIdStrings, not C++ enum names. These legacy
    # aliases come from constants/Events.h; normal names need only punctuation
    # normalization (e.g. "Big Fish" <-> BIG_FISH).
    aliases = {'Accursed Blacksmith': 'OMINOUS_FORGE', 'Addict': 'PLEADING_VAGRANT',
        'Back to Basics': 'ANCIENT_WRITING', 'Beggar': 'OLD_BEGGAR',
        'Fountain of Cleansing': 'THE_DIVINE_FOUNTAIN', 'Golden Wing': 'WING_STATUE',
        'Liars Game': 'THE_SSSSSERPENT', 'Nest': 'THE_NEST'}
    normalized = {_item_key(name): score for name, score in fixed.items()}
    normalized.update({_item_key(name): fixed[key] for name, key in aliases.items()})
    return normalized.get(_item_key(event))


def rank_overworld(state, legal, *, rmap=None, current_y=-1):
    """Return scored known actions; None means this screen needs fallback."""
    g, opts, phase = state['global'], state.get('options', {}), state['phase']
    if phase == 'combat':
        raise ValueError('OverworldRuleStrategy must be composed with a combat strategy')
    event = event_scores(state) if phase == 'event' and not opts.get('neow') else None
    if phase == 'event' and not opts.get('neow') and event is None:
        return None
    context = opts.get('select', {}).get('context')
    if phase == 'card_select' and context not in ('UPGRADE', 'REMOVE', 'TRANSFORM',
            'TRANSFORM_UPGRADE', 'DUPLICATE', 'OBTAIN', 'BOTTLE'):
        return None
    hp, max_hp = g['hp'], max(1, g['max_hp'])
    route = {}
    if phase == 'map':
        if rmap is None:
            return None
        choices = [(i, a) for a in legal for kind, i in [A.segment_of(a)] if kind == 'MAP_NODE']
        route = route_scores(state, choices, rmap, current_y)
    result = []
    for action in legal:
        kind, i = A.segment_of(action)
        score, reason = -100.0, kind.lower()
        if kind == 'USE_POTION':
            potion = _item_key(g.get('potions', ())[i])
            empty = sum(p in ('EMPTY_POTION_SLOT', 'INVALID', '') for p in g.get('potions', ()))
            score = 100 if potion == 'FRUITJUICE' else (100 if hp <= .8 * max_hp else -10) if potion == 'BLOODPOTION' else (90 if empty >= 2 else -10) if potion == 'ENTROPICBREW' else -100
        elif kind in ('REWARD_GOLD', 'REWARD_RELIC'):
            score = 80
        elif kind == 'REWARD_POTION':
            score = 70 if any(p in ('EMPTY_POTION_SLOT', 'INVALID', '') for p in g.get('potions', ())) else -10
        elif kind in ('PROCEED', 'REWARD_SKIP'):
            score = 0
        elif kind == 'REWARD_CARD':
            score = card_value(opts['cards'][i], state) - 5
        elif kind == 'CHEST':
            score = (10 if not has_relic(state, 'CURSED_KEY') else -5) if i == 0 else 0
        elif kind == 'NEOW':
            option = opts['neow'][i]
            bonus = [6, 5, 5, 5, 5, 5, 4, 8, 3, 6, 5, 9, 8, 9, 9, 8, 9, 5, 8]
            drawback = [0, 0, 3, 3, 7, 4, 7]
            b, d = option['bonus_id'], option['drawback_id']
            if b >= len(bonus) or d >= len(drawback):
                return None
            score = bonus[b] - drawback[d]
        elif kind == 'MAP_NODE':
            score = route[action]
        elif kind == 'PICK_CARD':
            card = opts['cards'][i]
            score = upgrade_value(card, state) if context == 'UPGRADE' else remove_value(card, state) if context in ('REMOVE', 'TRANSFORM', 'TRANSFORM_UPGRADE', 'BONFIRE_SPIRITS') else card_value(card, state, adding=context in ('OBTAIN', 'DUPLICATE'))
            reason = f'{context.lower()}:{card[0]}'
        elif kind == 'CAMPFIRE':
            name = A._CAMP_NAME_BY_IDX[i]
            score = {'rest': min(max_hp - hp, .3 * max_hp) * (1.1 if hp / max_hp < .55 else .4),
                'smith': max((upgrade_value(c, state) for c in g.get('deck', ())), default=0),
                'recall': -5, 'lift': 10, 'toke': max((remove_value(c, state) for c in g.get('deck', ())), default=0),
                'dig': 8}.get(name, -100)
            reason = name
        elif kind == 'BOSS_RELIC':
            score = relic_value(g['boss_relics'][i], state)
            reason = g['boss_relics'][i]
        elif kind == 'SHOP_CARD':
            cid, up, price = opts['shop_cards'][i]
            score = (card_value((cid, up), state) - 5) * 2 - price / 35
            reason = f'buy:{cid}'
        elif kind == 'SHOP_RELIC':
            relic, price = opts['shop_relics'][i]
            score = relic_value(relic, state) - price / 35
            reason = f'buy:{relic}'
        elif kind == 'SHOP_POTION':
            _, price = opts['shop_potions'][i]
            score = ((3 if hp / max_hp < .55 else 1) - price / 25
                if can_obtain_potion(state) else -100)
        elif kind == 'SHOP_REMOVE':
            score = max((remove_value(c, state) for c in g.get('deck', ())), default=0) - opts['shop_remove_cost'] / 20
        elif kind == 'EVENT_OPT' and event is not None:
            score = event.get(i, -100)
            reason = f'{opts["event"]}:{i}'
        result.append(dict(action=int(action), score=float(score), reason=reason))
    if not result or max(row['score'] for row in result) <= -100:
        return None
    return sorted(result, key=lambda row: (-row['score'], row['action']))


class OverworldRuleStrategy:
    name = 'overworld-rules'
    version = OVERWORLD_RULE_VERSION

    def __init__(self, fallback=None, *, trace=None):
        self.fallback, self.trace = fallback, trace

    def decide(self, requests, *, check=lambda: None):
        from rl.evaluation.strategies import Decision
        result, fallback_requests, fallback_indices, traces = [None] * len(requests), [], [], []
        for index, request in enumerate(requests):
            check()
            base = request.env.unwrapped.thick
            state = base.obs_canonical()
            legal = np.flatnonzero(request.env.action_masks())
            ranked = rank_overworld(state, legal, rmap=base.rmap,
                                    current_y=base.gc.cur_map_node_y)
            if ranked is None:
                if self.fallback is None:
                    raise ValueError(f'unsupported overworld screen without fallback: {state["phase"]}/{state.get("options", {})}')
                fallback_indices.append(index)
                fallback_requests.append(request)
            else:
                result[index] = Decision(ranked[0]['action'], self.name, {'overworld_rule_decisions': 1})
            if self.trace:
                traces.append(dict(kind='overworld_decision', seed=request.seed,
                    step=request.step, phase=state['phase'], version=self.version,
                    ranked=ranked, state=state))
        if fallback_requests:
            decisions = self.fallback.decide(fallback_requests, check=check)
            if len(decisions) != len(fallback_requests):
                raise ValueError('overworld fallback returned a wrong decision count')
            for index, decision in zip(fallback_indices, decisions):
                diagnostics = dict(decision.diagnostics)
                diagnostics['overworld_fallback_decisions'] = 1
                result[index] = Decision(decision.action, f'{self.name}:fallback:{decision.source}', diagnostics)
        if self.trace:
            for trace, decision in zip(traces, result):
                self.trace(dict(trace, action=decision.action, source=decision.source))
        return result
