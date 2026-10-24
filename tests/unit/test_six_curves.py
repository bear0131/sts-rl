"""Six HP prices ignore resource/outlet identity and all settlement increments."""
import copy
from types import SimpleNamespace

import numpy as np
import pytest

from env.core.finite_curves import (FiniteCurveBank, SIX_SCHEMA, SIX_PAYOFF,
                                  exit_identity, settled_exit, matched_exit_value)
from env.core.outer_return import CombatExit


@pytest.fixture
def bank():
    return FiniteCurveBank(dict(schema=SIX_SCHEMA, payoff=SIX_PAYOFF,
        teacher_sha256='1'*64, hp_fractions=[.01, .5, 1.],
        curves={f'{a}|{room}': ([1., 1., 1.] if (a, room) == (3, 'boss')
                else [a*.1, a*.1+.2, a*.1+.4])
                for a in (1, 2, 3) for room in ('boss', 'nonboss')}))


@pytest.mark.parametrize('act', [1, 2, 3])
@pytest.mark.parametrize('boss', [False, True])
def test_only_hp_act_and_room_affect_payoff(bank, act, boss):
    state = dict(act=act, is_boss=boss, combat_hp=40, combat_max_hp=80, outlet='win')
    expected = .5 if (act, boss) == (3, True) else (act*.1+.2)/2
    for outcome in ('win', 'escape'):
        for potions in range(6):
            changed = dict(state, outlet=outcome, potions=potions, hp=80, max_hp=100,
                           encounter='UNSEEN_ENCOUNTER', gold=10000, escaped_thieves=2)
            for reward in (-100., 0., 100.):
                assert matched_exit_value(CombatExit(reward, 'upper', changed), bank) == pytest.approx(expected)
    assert matched_exit_value(CombatExit(100., 'terminal', dict(state, outlet='loss')), bank) == 0.
    with pytest.raises(ValueError, match='observed'):
        matched_exit_value(CombatExit(0., 'censored', state), bank)


def test_capture_before_relic_or_boss_healing(bank):
    battle = SimpleNamespace(is_battle_over=True, to_dict=lambda: dict(
        outcome=1, player=dict(hp=8, max_hp=80), monsters=[]))
    gc = SimpleNamespace(act=1, encounter='SLIME_BOSS', cur_room='BOSS')
    state = exit_identity(gc, battle, include_combat_hp=True)
    gc.act, gc.cur_hp, gc.max_hp, gc.overworld_potions = 2, 80, 80, []
    state = settled_exit(state, gc)
    assert (state['act'], state['combat_hp'], state['combat_max_hp'], state['is_boss']) == (1, 8, 80, True)
    value = matched_exit_value(CombatExit(.2, 'upper', state), bank)
    assert value < matched_exit_value(CombatExit(.2, 'upper', dict(state, combat_hp=80)), bank)
    assert value == pytest.approx(np.interp(.1, [.01, .5, 1.], [.1, .3, .5])/2)


def test_exact_six_and_no_silent_clipping(bank):
    assert len(bank.curves) == 6 and bank.identity['schema'] == SIX_SCHEMA
    state = dict(act=3, is_boss=True, combat_hp=1, combat_max_hp=80, outlet='win')
    assert matched_exit_value(CombatExit(1.002, 'terminal', state), bank) == .5
    for change in (dict(combat_hp=81), dict(combat_max_hp=0), dict(is_boss=1), dict(act=4)):
        with pytest.raises(ValueError):
            bank.value(dict(state, **change))
    for mutate in ('missing', 'extra', 'potion', 'terminal', 'range'):
        data = copy.deepcopy(bank.data())
        if mutate == 'missing': data['curves'].pop('1|nonboss')
        if mutate == 'extra': data['curves']['1|escape'] = [.1, .2, .3]
        if mutate == 'potion': data['potion_counts'] = list(range(6))
        if mutate == 'terminal': data['curves']['3|boss'][0] = .5
        if mutate == 'range': data['curves']['1|boss'][0] = 2.1
        with pytest.raises(ValueError): FiniteCurveBank(data)
