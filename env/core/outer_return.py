"""TRAIN47's outer reward and the new combat continuation-value contract."""
from dataclasses import dataclass
import math

OBJECTIVE = 'combat-outer-return-frozen-upper-v1'
CONTEXT_SCHEMA = 'combat-public-run-context-v2'
CONTEXT_KEYS = frozenset(('act', 'floor', 'next_boss', 'run_hp', 'deck_ids',
                          'deck_upgrades', 'outer_deck_misc', 'character', 'ascension', 'run_keys',
                          'selected_card_ids', 'selected_card_upgrades'))
MISC_SCALE = 100.0


def permanent_misc(gc, battle=None):
    """Public permanent card growth, in the physical master-deck slots."""
    import numpy as np
    from .encode import DECK_SLOTS
    result = np.zeros(DECK_SLOTS, dtype=np.float32)
    deck = gc.deck_state()
    for i, card in enumerate(deck):
        result[i] = card['misc'] / MISC_SCALE
    if battle is not None:
        for index, value in battle.permanent_card_misc().items():
            if 0 <= index < len(deck):
                result[index] = value / MISC_SCALE
    return result


def strategic_reward(before_act, before_floor, stats, *, done, censored):
    """Exactly the original upper reward; raw environment death -1 is not used."""
    if censored:
        return 0.0
    reward = .002 * max(0, int(stats['floor']) - before_floor)
    for act, bonus in ((1, .1), (2, .2)):
        if before_act <= act < int(stats['act']):
            reward += bonus
    if done and (stats['act'] > 3 or stats['outcome'] == 'PLAYER_VICTORY'):
        reward += 1.0
    return reward


@dataclass(frozen=True)
class CombatExit:
    reward: float
    kind: str  # upper, combat, terminal, censored

    def target(self, next_value=None):
        if self.kind == 'censored':
            raise ValueError('censored combat has no supervised terminal target')
        if self.kind == 'terminal':
            if next_value is not None:
                raise ValueError('natural terminal must not bootstrap')
            result = self.reward
        elif self.kind in ('upper', 'combat'):
            if next_value is None:
                raise ValueError('continuation requires its next-state value')
            result = self.reward + float(next_value)
        else:
            raise ValueError(f'unknown combat exit kind: {self.kind}')
        if not math.isfinite(result):
            raise ValueError('outer return must be finite')
        return float(result)
