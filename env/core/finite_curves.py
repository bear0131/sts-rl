"""Finite frozen-teacher continuation curves; no model or simulator imports."""
from copy import deepcopy
import hashlib
import json
import math

import numpy as np

OBJECTIVE = 'combat-finite-upper-curves-v1'
FACTORED_OBJECTIVE = 'combat-finite-curves-win-conditional-v1'
CURVE_OBJECTIVES = (OBJECTIVE, FACTORED_OBJECTIVE)
SCHEMA = 'frozen-upper-hp-curve-bank-v1'


def curve_key(act, encounter, outlet):
    return f'{int(act)}|{encounter}|{outlet}'


def exit_identity(gc, battle):
    """Capture the original encounter before settlement can advance the game."""
    if battle is None or not battle.is_battle_over:
        return None
    from .combat_value import combat_terminal_outcome
    outcome = combat_terminal_outcome(battle)
    if outcome not in ('win', 'escape', 'loss'):
        raise ValueError('unknown finite-curve combat outcome')
    return dict(act=int(gc.act), encounter=str(gc.encounter).split('.')[-1], outlet=outcome)


def settled_exit(identity, gc):
    if identity is None:
        return None
    return dict(identity, hp=int(gc.cur_hp), max_hp=int(gc.max_hp),
                potions=sum(str(p) not in ('EMPTY_POTION_SLOT', 'INVALID')
                            for p in gc.overworld_potions))


def winning_exit_target(exit, bank, *, reward_prefix=0.):
    """Observed win label and conditional curve return, shared by search/learner.

    Player escape remains an explicit simulator outcome but is not victory.
    Natural game terminals never bootstrap; censored segments have no label.
    """
    if exit.kind == 'censored' or exit.curve_state is None:
        raise ValueError('winning curve target requires an observed combat exit')
    outcome = exit.curve_state['outlet']
    if outcome not in ('win', 'loss', 'escape'):
        raise ValueError('unknown winning curve outcome')
    if not math.isfinite(reward_prefix):
        raise ValueError('invalid reward prefix')
    if outcome != 'win':
        return 0., 0.
    bootstrap = None if exit.kind == 'terminal' else bank.value(exit.curve_state)
    return 1., exit.target(bootstrap) - reward_prefix


class FiniteCurveBank:
    """Immutable values embedded in the model constructor and replay identity.

    Curves vary post-settlement HP fraction and held potion count. Reference
    deck/relic/gold/reward identities are marginalized offline, never looked up
    from the live state. Interpolation is linear, with no monotonic projection.
    """
    def __init__(self, data):
        self._data = deepcopy(data)
        canonical = json.dumps(data, sort_keys=True, separators=(',', ':'), allow_nan=False)
        self.sha256 = hashlib.sha256(canonical.encode()).hexdigest()
        if data.get('schema') != SCHEMA:
            raise ValueError('unsupported finite curve bank schema')
        self.grid = np.asarray(data['hp_fractions'], dtype=np.float64)
        if (len(self.grid) < 2 or self.grid[0] <= 0 or self.grid[-1] != 1. or
                not np.isfinite(self.grid).all() or not np.all(np.diff(self.grid) > 0)):
            raise ValueError('invalid finite curve HP grid')
        if data.get('potion_counts') != list(range(6)):
            raise ValueError('finite curves require all six held-potion counts')
        self.curves = {key: np.asarray(value, dtype=np.float64) for key, value in data['curves'].items()}
        if not self.curves or any(x.shape != (6, len(self.grid)) or not np.isfinite(x).all()
                                  for x in self.curves.values()):
            raise ValueError('invalid finite curve values')
        self.aliases = dict(data.get('aliases', {}))
        if any(key in self.curves or value not in self.curves for key, value in self.aliases.items()):
            raise ValueError('invalid finite curve aliases')
        for array in (self.grid, *self.curves.values()):
            array.setflags(write=False)

    @property
    def identity(self):
        return dict(objective=OBJECTIVE, schema=SCHEMA, sha256=self.sha256,
                    teacher_sha256=self._data['teacher_sha256'])

    def data(self):
        return deepcopy(self._data)

    def value(self, state):
        # Natural losses never bootstrap. Censored outcomes must be discarded.
        if state['outlet'] == 'loss':
            return 0.
        if state['outlet'] not in ('win', 'escape'):
            raise ValueError('unknown finite curve outlet')
        hp, maximum, count = state['hp'], state['max_hp'], state['potions']
        if (not all(math.isfinite(float(x)) for x in (hp, maximum, count)) or
                maximum <= 0 or hp <= 0 or hp > maximum or int(count) != count or not 0 <= count <= 5):
            raise ValueError('invalid finite curve resources')
        key = curve_key(state['act'], state['encounter'], state['outlet'])
        key = self.aliases.get(key, key)
        if key not in self.curves:
            raise ValueError(f'finite curve bank has no encounter/outlet: {key}')
        return float(np.interp(hp/maximum, self.grid, self.curves[key][int(count)]))
