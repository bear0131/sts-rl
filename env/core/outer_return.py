"""TRAIN47's outer reward and the new combat continuation-value contract."""
from dataclasses import dataclass, replace
import math
from .finite_curves import OBJECTIVE as FINITE_OBJECTIVE

OBJECTIVE = 'combat-outer-return-frozen-upper-v1'
NORMALIZED_OBJECTIVE = 'combat-outer-return-remaining-max-v1'
MIXED_RAW_OBJECTIVE = 'combat-outer-return-raw-trajectory-mix-v1'
MC_OBJECTIVE = 'combat-actual-run-return-search-normalized-v1'
MC_EXIT_KEY = 'combat_mc_exit'
SCALAR_OBJECTIVES = (OBJECTIVE, FINITE_OBJECTIVE, NORMALIZED_OBJECTIVE, MIXED_RAW_OBJECTIVE, MC_OBJECTIVE)
NORMALIZATION = dict(schema='remaining-strategic-reward-max-v1', last_floor=51,
                     ascension=0, last_act=3, clip=False)
CONTEXT_SCHEMA = 'combat-public-run-context-v2'
CONTEXT_KEYS = frozenset(('act', 'floor', 'next_boss', 'run_hp', 'deck_ids',
                          'deck_upgrades', 'outer_deck_misc', 'character', 'ascension', 'run_keys',
                          'selected_card_ids', 'selected_card_upgrades'))
MISC_SCALE = 100.0


def remaining_reward_max(act, floor, *, terminal=False):
    """A0 Act1–3 reward ceiling at the source state, before settlement.

    Floor progress ends at the Act3 victory room (51); Act1/2 completion
    bonuses and the final win remain until settled. This is not a clamp
    on learned predictions, which may be outside the target range.
    """
    if terminal:
        return 0.
    if act not in (1, 2, 3) or not 0 <= floor <= 50:
        raise ValueError('remaining reward normalization requires live A0 Act1–3 state')
    return .002 * (51 - floor) + (.1 if act == 1 else 0.) + (.2 if act <= 2 else 0.) + 1.


def observation_reward_max(obs, index):
    from .encode import ACT_SCALE, FLOOR_SCALE
    return remaining_reward_max(round(float(obs['act'][index, 0]) * ACT_SCALE),
                                round(float(obs['floor'][index, 0]) * FLOOR_SCALE))


def normalized_exit(outcome, before_act, before_floor, after_act, after_floor):
    """Convert settled raw reward and continuation into the source units."""
    return replace(outcome, value_scale=remaining_reward_max(before_act, before_floor),
                   continuation_scale=(remaining_reward_max(after_act, after_floor)
                                       if outcome.kind == 'combat' else 1.))


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
    curve_state: dict | None = None
    value_scale: float = 1.
    continuation_scale: float = 1.

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
            result = self.reward + float(next_value) * self.continuation_scale
        else:
            raise ValueError(f'unknown combat exit kind: {self.kind}')
        if not (math.isfinite(self.value_scale) and self.value_scale > 0 and
                math.isfinite(self.continuation_scale) and self.continuation_scale > 0):
            raise ValueError('outer return scales must be finite and positive')
        result /= self.value_scale
        if not math.isfinite(result):
            raise ValueError('outer return must be finite')
        return float(result)
