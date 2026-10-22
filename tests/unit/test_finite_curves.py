from copy import deepcopy
from types import SimpleNamespace
import numpy as np
import pytest

from env.core.finite_curves import FiniteCurveBank, SCHEMA, settled_exit


def bank_data():
    return dict(schema=SCHEMA, teacher_sha256='1'*64, hp_fractions=[.01, .5, 1.],
                potion_counts=list(range(6)), curves={
                    '2|TWO_THIEVES|win': [[.1+p*.01, .3+p*.01, .2+p*.01] for p in range(6)],
                    '2|TWO_THIEVES|escape': [[.01, .02, .03] for _ in range(6)]})


def test_curve_interpolation_retains_nonmonotonic_teacher_and_distinct_escape():
    bank = FiniteCurveBank(bank_data())
    state = dict(act=2, encounter='TWO_THIEVES', outlet='win', hp=60, max_hp=80, potions=2)
    assert bank.value(state) == pytest.approx(.27)
    assert bank.value(dict(state, outlet='escape')) == pytest.approx(.025)
    assert bank.value(dict(state, outlet='loss', hp=0)) == 0
    # Runtime deck/relic/gold and potion types have no lookup influence.
    assert bank.value(dict(state, deck='different', relics=['X'], gold=-100)) == bank.value(state)
    assert bank.value(dict(state, hp=80)) < bank.value(dict(state, hp=40))


def test_bank_rejects_missing_groups_invalid_resources_and_mutable_source():
    data = bank_data(); bank = FiniteCurveBank(data)
    identity = bank.identity; data['curves']['2|TWO_THIEVES|win'][0][0] = 999
    assert bank.data() != data and bank.identity == identity
    changed = bank_data(); changed['curves']['2|TWO_THIEVES|win'][0][0] += .01
    assert FiniteCurveBank(changed).identity != identity
    state = dict(act=2, encounter='TWO_THIEVES', outlet='win', hp=30, max_hp=80, potions=0)
    for diff in (dict(potions=6), dict(potions=1.5), dict(hp=0), dict(hp=np.nan), dict(outlet='unknown'), dict(encounter='MISSING')):
        with pytest.raises(ValueError):bank.value(dict(state, **diff))
    bad = bank_data(); bad['hp_fractions']=[0,.5,1]
    with pytest.raises(ValueError):FiniteCurveBank(bad)


def test_settled_resource_count_preserves_physical_empty_slots():
    gc = SimpleNamespace(cur_hp=35,max_hp=80,overworld_potions=['EMPTY_POTION_SLOT','Fire Potion','EMPTY_POTION_SLOT','Blood Potion'])
    state = settled_exit(dict(act=2, encounter='TWO_THIEVES', outlet='win'),gc)
    assert state['potions'] == 2 and state['hp'] == 35
