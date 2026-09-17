"""Visible-state Ironclad heuristics. No network, transition oracle or RNG access.

Scores are preferences, not predicted HP/Q. See README.md for information and
comparison boundaries. Rule version changes when scoring behavior changes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np

from env.core import action_space as A, actions, game_actions as ga
from env.core.ironclad_cards import IRONCLAD_CARDS, card_numbers
from schemas.combat_state import normalize_sim, _norm_sim_monster, _sim_card_id

RULE_VERSION = 'ironclad-v1'
BAD_CARDS = frozenset(('Wound', 'Dazed', 'Burn', 'Void', 'Slimed', 'AscendersBane',
    'Injury', 'Doubt', 'Shame', 'Pain', 'Regret', 'Decay', 'Normality', 'Clumsy',
    'CurseOfTheBell', 'Parasite', 'Writhe', 'Pride', 'Necronomicurse'))
AOE = frozenset(('Cleave', 'Thunderclap', 'Whirlwind', 'Immolate', 'Reaper'))
DRAW = {'Pommel Strike': (1, 2), 'Shrug It Off': (1, 1), 'Warcry': (1, 2),
    'Battle Trance': (3, 4), 'Burning Pact': (2, 3), 'Offering': (3, 5),
    'Flash Of Steel': (1, 1), 'Finesse': (1, 1), 'Master Of Strategy': (3, 4)}
# Common neutral/event cards: (cost, damage, block). Unknown cards are diagnosed.
EXTRA = {'Bite': (1, 7, 0), 'Flash Of Steel': (0, 3, 0), 'Finesse': (0, 0, 2),
    'Swift Strike': (0, 7, 0), 'Good Instincts': (0, 0, 6),
    'Panacea': (0, 0, 0), 'Trip': (0, 0, 0), 'Blind': (0, 0, 0),
    'Master Of Strategy': (0, 0, 0), 'Panache': (0, 0, 0),
    'Discovery': (1, 0, 0), 'Secret Weapon': (0, 0, 0),
    'Secret Technique': (0, 0, 0), 'Apotheosis': (2, 0, 0),
    'Dramatic Entrance': (0, 8, 0), 'Bandage Up': (0, 0, 0)}
SELECT_PILE = {**dict.fromkeys(('HEADBUTT', 'HOLOGRAM', 'LIQUID_MEMORIES_POTION', 'MEDITATE'), 'discard'),
    'EXHUME': 'exhaust', **dict.fromkeys(('SECRET_TECHNIQUE', 'SECRET_WEAPON', 'SEEK'), 'draw'),
    **dict.fromkeys(('CODEX', 'DISCOVERY'), 'generated'),
    **dict.fromkeys(('ARMAMENTS', 'DUAL_WIELD', 'EXHAUST_ONE', 'EXHAUST_MANY',
                    'FORETHOUGHT', 'GAMBLE', 'NIGHTMARE', 'RECYCLE', 'SETUP', 'WARCRY'), 'hand')}


@dataclass
class CombatView:
    player: dict
    monsters: list  # None occupies an empty native slot; never compact this list.
    hand: list
    piles: dict
    potions: list = field(default_factory=list)
    relics: dict = field(default_factory=dict)
    select: dict = field(default_factory=dict)
    pending: tuple | None = None
    targets: dict = field(default_factory=dict)
    turn: int = 1

    @property
    def alive(self):
        return [m for m in self.monsters if m and m['hp'] > 0 and not m.get('gone')]

    @property
    def incoming(self):
        return sum(intent(m) for m in self.alive)

    @property
    def deficit(self):
        return max(0, self.incoming - self.player['block'] - self.player['powers'].get('Metallicize', 0))


def visible_combat(request):
    """Simulator adapter; preserves slot identity and hides ordinary draw order.

    A draw-pile SELECT screen is the only point where draw indices are exposed:
    they identify visible choices, never used to predict the next draw.
    """
    env = request.env.unwrapped
    if request.info.get('phase') != 'combat' or env.thick.bc is None:
        raise ValueError('combat rules require the simulator combat backend')
    raw = env.thick.bc.to_dict()
    canon = normalize_sim(raw)
    monsters = [None if m['id'] == 'INVALID = 0' else _norm_sim_monster(m) for m in raw['monsters']]
    select = dict(env.thick.select_descriptor() or {})
    generated = []
    if select.get('context') in ('CODEX', 'DISCOVERY'):
        info = env.thick.bc.select_info()
        if 'generated_cards' not in info:
            raise RuntimeError('combat rules require a native build exposing select_info.generated_cards')
        generated = [{'id': _sim_card_id(c[0]), 'up': c[1]} for c in info['generated_cards']]
    piles = {key: canon[key] for key in ('draw', 'discard', 'exhaust')}
    if SELECT_PILE.get(select.get('context')) != 'draw':
        piles['draw'] = sorted(piles['draw'], key=lambda c: (c['id'], c['up']))
    piles['generated'] = generated
    targets = {}
    for semantic in env.thick.legal_semantic_actions():
        if semantic[0] == ga.COMBAT:
            decoded = actions.decode(semantic[1])
            if decoded[0] == actions.KIND_PLAY:
                targets.setdefault(('play', decoded[1]), []).append(decoded[2])
        elif semantic[0] == ga.USE_POTION:
            targets.setdefault(('potion', semantic[1]), []).append(semantic[2])
    return CombatView(canon['player'], monsters, canon['hand'], piles,
        list(raw['potions']), {r['id']: r['counter'] for r in env.thick.gc.relic_state()},
        select, request.info.get('pending'), targets, canon['turn'])


def intent(monster):
    return max(0, monster.get('adjusted_damage', 0)) * max(0, monster.get('hits', 0))


def numbers(card):
    result = card_numbers(card['id'], card.get('up', 0))
    return result if result is not None else EXTRA.get(card['id'], (max(0, card.get('cost', 1)), 0, 0))


def kind(card):
    info = IRONCLAD_CARDS.get(card['id'])
    return info.kind if info else 'attack' if numbers(card)[1] else 'skill'


def cost(card, view):
    value = card.get('cost', numbers(card)[0])
    if kind(card) == 'skill' and view.player['powers'].get('Corruption', 0):
        value = 0
    return max(0, view.player['energy'] if value == -1 else value)


def damage(card, monster, view):
    """Hand calculation estimate; does not call the simulator damage/step API."""
    cid, up = card['id'], bool(card.get('up'))
    p, powers = view.player, view.player['powers']
    base = numbers(card)[1]
    strength = powers.get('Strength', 0)
    if cid == 'Body Slam':
        base = p['block']
    elif cid == 'Heavy Blade':
        base += strength * (4 if up else 2)
    elif cid == 'Perfected Strike':
        base += (3 if up else 2) * sum('Strike' in c['id'] for c in view.hand + view.piles['draw'] + view.piles['discard'])
    hits = {'Twin Strike': 2, 'Pummel': 5 if up else 4,
        'Sword Boomerang': 4 if up else 3, 'Whirlwind': p['energy'],
        'Fiend Fire': max(0, len(view.hand)-1)}.get(cid, 1)
    if cid == 'Whirlwind' and 'Chemical X' in view.relics:
        hits += 2
    value = max(0, base + strength + powers.get('Vigor', 0))
    if 'Strike Dummy' in view.relics and 'Strike' in cid:
        value += 3
    if powers.get('Weakened', 0):
        value *= .75
    if powers.get('Double Damage', 0) or powers.get('DoubleDamage', 0):
        value *= 2
    if view.player.get('combat_counters', {}).get('pen_nib', 0) == 9:
        value *= 2
    mp = monster['powers']
    if mp.get('Vulnerable', 0):
        value *= 1.75 if 'Paper Frog' in view.relics else 1.5
    value *= 1 + .1 * mp.get('Slow', 0)
    if mp.get('Flight', 0):
        value *= .5
    value = math.floor(value)
    if mp.get('Intangible', 0):
        value = min(1, value)
    if cid == 'Sword Boomerang' and len(view.alive) > 1:
        # Expected damage, never pretend random hits guarantee a lethal.
        hits /= len(view.alive)
    if powers.get('Double Tap', 0) or powers.get('DoubleTap', 0):
        hits *= 2
    return value * hits


def block_value(amount, view):
    useful = min(amount, view.deficit)
    carry = max(0, amount-useful) * (.6 if view.player['powers'].get('Barricade', 0) else .02)
    danger = 2.5 if view.deficit >= view.player['hp'] else 1.35
    return useful * danger + carry


def card_score(card, target, view):
    cid, up = card['id'], int(bool(card.get('up')))
    p, pp = view.player, view.player['powers']
    remaining = max(0, p['energy'] - cost(card, view))
    alive = view.alive
    target_monster = view.monsters[target] if target is not None else None
    affected = alive if cid in AOE or cid in ('Sword Boomerang', 'Dramatic Entrance') else ([target_monster] if target_monster else [])
    attack = kind(card) == 'attack'
    parts = {}
    damage_total, broken_block, kills, avoided, retaliation = 0., 0., 0, 0., 0.
    if attack:
        for m in affected:
            raw_damage = damage(card, m, view)
            broken_block += min(raw_damage, m['block'])
            dealt = max(0, raw_damage - m['block'])
            damage_total += min(m['hp'], dealt)
            guaranteed = cid != 'Sword Boomerang' or len(alive) == 1
            if dealt >= m['hp'] and guaranteed:
                kills += 1
                avoided += intent(m)
            retaliation += m['powers'].get('Thorns', 0) + m['powers'].get('Sharp Hide', 0)
        parts['damage'] = .85 * damage_total
        parts['break_block'] = .65 * broken_block
        parts['kill'] = 22 * kills + 1.5 * avoided
        if alive and kills == len(alive):
            parts['combat_lethal'] = 10000
        if cid in ('Reaper', 'Bite'):
            heal = damage_total if cid == 'Reaper' else 3 if up else 2
            parts['heal'] = 2 * min(p['max_hp']-p['hp'], heal) if not pp.get('NoHealing', 0) else 0
        if cid == 'Feed' and kills:
            parts['max_hp'] = 18
        parts['retaliation'] = -1.5 * max(0, retaliation-p['block'])
    block = numbers(card)[2]
    if cid == 'Entrench':
        block = p['block']  # not modified by Dexterity/Frail
    if cid == 'Second Wind':
        block = (7 if up else 5) * sum(kind(c) != 'attack' for c in view.hand if c is not card)
    if block and cid != 'Entrench':
        block = max(0, block + pp.get('Dexterity', 0))
        if pp.get('Frail', 0):
            block = math.floor(.75 * block)
    if pp.get('NoBlock', 0):
        block = 0
    parts['block'] = block_value(block, view)
    if cid == 'Flame Barrier':
        parts['counterattack'] = (6 if up else 4) * sum(m['hits'] for m in alive if intent(m)) * .7
    attacks = [c for c in view.hand if c is not card and kind(c) == 'attack']
    affordable_attacks = min(len(attacks), remaining + sum(cost(c, view) == 0 for c in attacks))
    if cid in ('Bash', 'Thunderclap', 'Shockwave', 'Uppercut', 'Trip'):
        victims = alive if cid in ('Thunderclap', 'Shockwave') or (cid == 'Trip' and up) else ([target_monster] if target_monster else [])
        parts['vulnerable'] = sum(4 + 3 * affordable_attacks for m in victims if not m['powers'].get('Vulnerable', 0) and not m['powers'].get('Artifact', 0))
    if cid in ('Clothesline', 'Uppercut', 'Shockwave', 'Intimidate', 'Blind'):
        victims = alive if cid in ('Shockwave', 'Intimidate') or (cid == 'Blind' and up) else ([target_monster] if target_monster else [])
        parts['weak'] = sum(3 + .35 * intent(m) for m in victims if not m['powers'].get('Weakened', 0) and not m['powers'].get('Artifact', 0))
    if cid == 'Disarm' and target_monster and not target_monster['powers'].get('Artifact', 0):
        parts['strength_down'] = (3 if up else 2) * (2.5 + 1.5 * target_monster['hits'])
    if cid in DRAW:
        draw = min(DRAW[cid][up], 11-len(view.hand), len(view.piles['draw'])+len(view.piles['discard']))
        parts['draw'] = 0 if pp.get('No Draw', 0) or pp.get('NoDraw', 0) else draw * (3.5 if remaining > 0 else .6)
    if cid == 'Dropkick' and target_monster and target_monster['powers'].get('Vulnerable', 0):
        parts['refund_and_draw'] = 9
    if cid in ('Seeing Red', 'Bloodletting', 'Offering'):
        extra = 3 if cid == 'Bloodletting' and up else 2
        demand = sum(cost(c, view) for c in view.hand if c is not card and c['id'] not in BAD_CARDS)
        parts['energy'] = 5 * min(extra, max(0, demand-remaining))
    hp_cost = {'Offering': 6, 'Bloodletting': 3, 'Hemokinesis': 2}.get(cid, 0)
    if hp_cost:
        parts['hp_cost'] = -hp_cost * (1000 if hp_cost >= p['hp'] else 1.7)
    if cid in ('Flex', 'Inflame', 'Spot Weakness', 'Limit Break'):
        gain = {'Flex': 4 if up else 2, 'Inflame': 3 if up else 2,
                'Spot Weakness': 4 if up else 3, 'Limit Break': max(0, pp.get('Strength', 0))}[cid]
        if cid == 'Spot Weakness' and (not target_monster or not intent(target_monster)):
            gain = 0
        parts['strength'] = gain * (affordable_attacks + (0 if cid == 'Flex' else 2.5))
        if cid == 'Flex' and affordable_attacks and gain:
            # Free setup must precede spending the attacks it improves.
            parts['setup_order'] = 8
    horizon = min(4, max(1, sum(m['hp'] for m in alive) / 35))
    all_cards = view.hand + view.piles['draw'] + view.piles['discard']
    exhaust_count = sum(c['id'] in BAD_CARDS or c['id'] in ('True Grit', 'Burning Pact', 'Second Wind', 'Fiend Fire', 'Offering', 'Warcry') for c in all_cards)
    power_values = {'Metallicize': (4 if up else 3) * horizon, 'Demon Form': (3 if up else 2) * horizon * 2,
        'Combust': (7 if up else 5) * len(alive) * horizon - 2,
        'Feel No Pain': (4 if up else 3) * min(4, exhaust_count),
        'Dark Embrace': 3 * min(4, exhaust_count),
        'Evolve': 3 * sum(c['id'] in BAD_CARDS for c in all_cards),
        'Fire Breathing': (10 if up else 6) * sum(c['id'] in BAD_CARDS for c in all_cards),
        'Barricade': 7 if any(c['id'] in ('Entrench', 'Impervious', 'Body Slam') for c in all_cards) else 1,
        'Corruption': 4 * sum(kind(c) == 'skill' and cost(c, view) > 0 for c in view.hand if c is not card),
        'Berserk': 4 * horizon - view.deficit * .6,
        'Brutality': 2 * horizon, 'Rupture': 4 * sum(c['id'] in ('Offering', 'Bloodletting', 'Hemokinesis', 'Combust') for c in all_cards),
        'Juggernaut': 8, 'Panache': 6}
    if cid in power_values:
        parts['power'] = power_values[cid]
    if cid == 'Rage':
        parts['attack_block'] = block_value((5 if up else 3)*affordable_attacks, view)
    if cid == 'Armaments':
        parts['upgrade'] = (2.5 * sum(not c.get('up') and c['id'] not in BAD_CARDS for c in view.hand if c is not card)) if up else 2.5
    if cid == 'Double Tap':
        parts['double_attack'] = max((numbers(c)[1] for c in attacks if cost(c, view) <= remaining), default=0) * .8
    if cid in ('Exhume', 'Headbutt', 'Secret Weapon', 'Secret Technique', 'Dual Wield', 'Discovery', 'Infernal Blade', 'Havoc'):
        parts['card_access'] = 3 if remaining else .5
    if cid == 'Apotheosis':
        parts['upgrade_deck'] = 2 * sum(not c.get('up') and c['id'] not in BAD_CARDS for c in all_cards)
    if cid == 'Bandage Up':
        parts['heal'] = 2 * min(p['max_hp']-p['hp'], 6 if up else 4)
    if cid == 'Panacea':
        parts['artifact'] = 2
    if cid in ('True Grit', 'Burning Pact', 'Second Wind'):
        parts['remove_junk'] = 3 * sum(c['id'] in BAD_CARDS for c in view.hand)
    if cid == 'Fiend Fire' and not parts.get('combat_lethal'):
        parts['exhaust_hand'] = -2 * sum(c['id'] not in BAD_CARDS and c is not card for c in view.hand)
    if cid in ('Wild Strike', 'Reckless Charge', 'Power Through', 'Immolate'):
        parts['add_junk'] = -1.5
    if not attack:
        parts['nob_enrage'] = -sum(2.5*m['powers'].get('Anger', 0) for m in alive) if kind(card) == 'skill' else 0
    parts['beat_of_death'] = -sum(m['powers'].get('BeatOfDeath', m['powers'].get('Beat of Death', 0)) for m in alive)
    if cid not in IRONCLAD_CARDS and cid not in EXTRA:
        parts['unknown_card'] = -1 if cid not in BAD_CARDS else -5
    # Prefer efficient cards, but a guaranteed kill takes precedence over cost.
    total = sum(parts.values())
    if not parts.get('kill') and not parts.get('combat_lethal'):
        total /= max(1, cost(card, view))
    total -= .25  # An effect-free play should lose to End Turn.
    return total, parts


def potion_score(slot, target, view):
    name = view.potions[slot].lower().replace('_', ' ')
    m = view.monsters[target] if target is not None else None
    value = 0.
    reserve = 13.
    if 'fire potion' in name and m:
        dealt = max(0, 20-m['block'])
        value = .85*min(dealt, m['hp']) + (30+intent(m) if dealt >= m['hp'] else 0)
    elif 'explosive' in name:
        value = sum(.85*min(10, m['hp']) + (20+intent(m) if 10 >= m['hp']+m['block'] else 0) for m in view.alive)
    elif 'block' in name:
        value = block_value(12, view)
    elif 'weak' in name and m and not m['powers'].get('Artifact', 0):
        value = 6 + .5*intent(m)
    elif 'fear' in name and m and not m['powers'].get('Artifact', 0):
        value = 5 + 4*view.player['energy']
    elif 'strength' in name:
        value = 12 if view.turn == 1 else 6
    elif 'dexterity' in name:
        value = 10 if view.turn == 1 else 5
    elif 'energy' in name:
        value = 6*min(2, max(0, sum(cost(c, view) for c in view.hand)-view.player['energy']))
    elif 'blood' in name:
        value = 2*min(view.player['max_hp']-view.player['hp'], view.player['max_hp']*.2)
    elif 'swift' in name or 'attack' in name or 'skill' in name or 'power' in name or 'colorless' in name:
        value = 14 if view.player['energy'] > 0 and len(view.hand) < 7 else 0
    elif 'ghost' in name:
        value = 1.5*view.deficit
    elif 'liquid memories' in name:
        value = 16 if view.piles['discard'] and view.deficit > 10 else 0
    elif 'poison' in name and m:
        value = 12 if not m['powers'].get('Artifact', 0) else 0
    emergency = view.deficit >= view.player['hp']
    if emergency:
        reserve = 0
    elif view.player['hp'] < view.player['max_hp']*.4:
        reserve = 7
    return value-reserve, {'potion_benefit': value, 'potion_reserve': -reserve}


def selection_score(index, view):
    context = view.select.get('context')
    if context not in SELECT_PILE:
        raise ValueError(f'unsupported combat selection context: {context}')
    pile = SELECT_PILE[context]
    candidates = view.hand if pile == 'hand' else view.piles[pile]
    if context == 'CODEX' and index == 3:
        return 2., {'skip_generated': 2.}
    if not 0 <= index < len(candidates):
        raise ValueError(f'{context} candidate {index} missing from {pile}')
    card = candidates[index]
    targets = [i for i, m in enumerate(view.monsters) if m and m['hp'] > 0 and not m.get('gone')]
    score = max((card_score(card, t, view)[0] for t in targets), default=0)
    score = min(35, score)  # return/upgrade value is not an immediate lethal
    if card['id'] in BAD_CARDS:
        score = -12
    if context in ('EXHAUST_ONE', 'EXHAUST_MANY', 'RECYCLE'):
        score = -score + (6 if card['id'] == 'Sentinel' else 0)
    elif context in ('WARCRY', 'FORETHOUGHT', 'SETUP', 'GAMBLE'):
        score = -score + (6 if cost(card, view) > view.player['energy'] else 0)
    elif context == 'ARMAMENTS':
        original = score
        upgraded = dict(card, up=1)
        score = max((card_score(upgraded, t, view)[0] for t in targets), default=0)-original + .1*original
    return score, {'select_'+context.lower()+'_'+pile: score}


def rank_actions(view, legal):
    ranked = []
    for action in legal:
        segment, index = A.segment_of(int(action))
        target = None
        if segment == 'END_TURN':
            score, reasons = 0., {'end_turn': 0.}
        elif segment == 'CONFIRM_SELECT':
            score, reasons = 0., {'confirm_selection': 0.}
        elif segment == 'PICK_CARD':
            score, reasons = selection_score(index, view)
        elif segment in ('PLAY_CARD', 'USE_POTION', 'PICK_TARGET'):
            source, slot = view.pending if segment == 'PICK_TARGET' else ('play' if segment == 'PLAY_CARD' else 'potion', index)
            choices = [index] if segment == 'PICK_TARGET' else view.targets.get((source, slot), [None])
            scored = [(card_score(view.hand[slot], t, view) if source == 'play' else potion_score(slot, t, view), t) for t in choices]
            (score, reasons), target = max(scored, key=lambda item: item[0][0])
        else:
            raise ValueError(f'noncombat action passed to combat rules: {segment}')
        ranked.append(dict(action=int(action), score=round(float(score), 6), target=target,
                           reasons={k: round(float(v), 4) for k, v in reasons.items() if v}))
    return sorted(ranked, key=lambda x: (-x['score'], x['action']))


class CombatRuleStrategy:
    name = 'rules-'+RULE_VERSION

    def __init__(self, *, trace=None):
        self.trace = trace

    def decide(self, requests, *, check=lambda: None):
        from rl.evaluation.strategies import Decision
        decisions = []
        for request in requests:
            check()
            view = visible_combat(request)
            ranked = rank_actions(view, np.flatnonzero(request.env.action_masks()))
            if not ranked:
                raise ValueError('combat rule received no legal action')
            best = ranked[0]
            unknown = int('unknown_card' in best['reasons'])
            unknown_options = sum('unknown_card' in option['reasons'] for option in ranked)
            decisions.append(Decision(best['action'], self.name,
                {'rule_decisions': 1, 'rule_unknown_choices': unknown,
                 'rule_unknown_options': unknown_options}))
            if self.trace:
                self.trace(dict(seed=request.seed, step=request.step, turn=view.turn,
                    player=view.player, monsters=view.monsters, hand=view.hand,
                    select=view.select, pending=view.pending, ranked=ranked))
        return decisions
