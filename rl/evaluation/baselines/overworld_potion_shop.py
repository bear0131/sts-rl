"""One affordable combat-potion reserve after v1's preferred shop purchases.

Only Act2, empty belts and the original v1 PROCEED decision are supplemented.
Effects and prices are public; scores are human priors, not expected saved HP.
No battle/encounter rollouts, future stock or private map outcomes are queried.
"""
import numpy as np

from env.core import action_space as A
from env.core.ironclad_cards import IRONCLAD_CARDS
from rl.evaluation.baselines.overworld_rules import (
    EMPTY_POTION_SLOTS as EMPTY_SLOTS, OverworldRuleStrategy, _item_key,
    can_obtain_potion, rank_overworld,
)
from rl.evaluation.strategies import Decision

OVERWORLD_POTION_SHOP_VERSION = 'ironclad-overworld-potion-shop-v1'
FAST_AOE = frozenset(('Cleave', 'Whirlwind', 'Immolate', 'Thunderclap'))
MULTIHIT = frozenset(('Twin Strike', 'Sword Boomerang', 'Pummel', 'Heavy Blade', 'Whirlwind'))


def potion_preference(potion, state):
    """Small native-verified whitelist; unknown/random/generated effects abstain."""
    potion = _item_key(potion)
    deck = state['global'].get('deck', ())
    attacks = sum(c in IRONCLAD_CARDS and IRONCLAD_CARDS[c].kind == 'attack' for c, _ in deck)
    nonstarter_attacks = sum(c in IRONCLAD_CARDS and IRONCLAD_CARDS[c].kind == 'attack'
                            and c not in ('Strike_R', 'Bash') for c, _ in deck)
    blocks = sum(c in IRONCLAD_CARDS and IRONCLAD_CARDS[c].block[bool(u)] > 0 for c, u in deck)
    improved_blocks = sum(c in IRONCLAD_CARDS and IRONCLAD_CARDS[c].block[bool(u)] > 0
                          and c != 'Defend_R' for c, u in deck)
    multi = sum(c in MULTIHIT for c, _ in deck)
    if potion == 'FIREPOTION':
        return 4 + int(nonstarter_attacks < 3), 'focused-damage'
    if potion == 'EXPLOSIVEPOTION':
        return 4 + 2 * int(not any(c in FAST_AOE for c, _ in deck)), 'area-damage'
    if potion == 'BLOCKPOTION':
        return 4 + int(improved_blocks < 2), 'immediate-block'
    if potion == 'WEAKPOTION':
        return 4, 'enemy-weak'
    if potion == 'FEARPOTION' and attacks >= 3:
        return 4 + int(multi > 0), 'attack-debuff'
    if potion == 'STRENGTHPOTION' and attacks >= 3:
        return 4 + min(2, multi), 'attack-scaling'
    if potion == 'DEXTERITYPOTION' and blocks >= 3:
        return 4 + int(blocks >= 6), 'block-scaling'
    if potion == 'HEARTOFIRON':
        return 6, 'recurring-block'
    if potion == 'ESSENCEOFSTEEL':
        return 5, 'depleting-block'
    return None


def rank_potion_shop(state, legal):
    g = state['global']
    if state['phase'] != 'shop' or g.get('act') != 2:
        return None
    held = g.get('potions', ())
    # INVALID is not an obtainable empty slot; honor the actual exposed belt.
    if not held or any(p not in EMPTY_SLOTS for p in held):
        return None
    if not can_obtain_potion(state):
        return None
    ranked = rank_overworld(state, legal)
    if ranked is None or A.segment_of(ranked[0]['action'])[0] != 'PROCEED':
        return None
    original = ranked[0]['action']
    changed = False
    for row in ranked:
        kind, index = A.segment_of(row['action'])
        if kind != 'SHOP_POTION':
            continue
        potion, price = state['options']['shop_potions'][index]
        if price < 0 or price > g['gold'] * .5 or g['gold'] - price < 75:
            continue
        preference = potion_preference(potion, state)
        if preference is None:
            continue
        priority, role = preference
        score = priority - price / 25
        if score <= 0:
            continue
        row['v1_score'] = row['score']
        row['score'] = score
        row['reason'] = f'prepare:{potion}'
        row['preparation'] = dict(potion=potion, role=role, priority=priority,
            price=price, gold_after=g['gold'] - price, price_fraction=price / max(1, g['gold']))
        changed = True
    if not changed:
        return None
    ranked.sort(key=lambda row: (-row['score'], row['action']))
    return ranked, original


class PotionShopUpperStrategy:
    name = 'overworld-potion-shop'
    version = OVERWORLD_POTION_SHOP_VERSION

    def __init__(self, fallback=None, *, trace=None):
        self.fallback, self.trace = fallback, trace
        self._base = OverworldRuleStrategy(fallback, trace=trace)

    def manifest(self):
        return dict(scope='act2-shop-only-when-v1-would-proceed', base_version=self._base.version,
            inventory='one-reserve-empty-belt-only', max_gold_fraction=.5, minimum_gold_after=75,
            preference='native-effect-whitelist-public-deck-role-minus-displayed-price',
            purchase_guards='actual-empty-slot-no-sozu-no-second-reserve',
            combat_use='unchanged-lower-policy-no-elite-preservation-guarantee')

    def decide(self, requests, *, check=lambda: None):
        result, delegated, indices = [None] * len(requests), [], []
        for index, request in enumerate(requests):
            check()
            state = request.env.unwrapped.thick.obs_canonical()
            ranked = rank_potion_shop(state, np.flatnonzero(request.env.action_masks()))
            if ranked is None:
                delegated.append(request)
                indices.append(index)
                continue
            rows, original = ranked
            action = rows[0]['action']
            result[index] = Decision(action, self.name, dict(overworld_rule_decisions=1,
                overworld_potion_shop_decisions=1, overworld_potion_shop_changed_decisions=int(action != original)))
            if self.trace:
                self.trace(dict(kind='overworld_decision', seed=request.seed, step=request.step,
                    phase=state['phase'], version=self.version, ranked=rows, state=state,
                    action=action, source=self.name, v1_action=original))
        if delegated:
            decisions = self._base.decide(delegated, check=check)
            for index, decision in zip(indices, decisions):
                result[index] = decision
        return result
