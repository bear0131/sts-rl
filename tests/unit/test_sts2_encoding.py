from copy import deepcopy

import numpy as np
import pytest

from env.sts2.encoding import features, visible_combat

pytestmark = pytest.mark.sts2


def test_hidden_state_and_draw_order_do_not_change_observation():
    raw = dict(player=dict(hp=80, energy=3, private_rng=99), rng=42,
               encounter=[dict(kind='a', hp=0, private_ai=3), dict(kind='b', hp=20)],
               deck=dict(hand=[dict(id='Strike', stable_idx=1)],
                         draw_pile=[dict(id='Defend', stable_idx=2), dict(id='Bash', stable_idx=3)]))
    other = deepcopy(raw)
    other['rng'] = 999
    other['player']['private_rng'] = 100
    other['encounter'][0]['private_ai'] = 999
    other['deck']['draw_pile'].reverse()
    other['deck']['hand'][0]['stable_idx'] = 99
    visible = visible_combat(raw)
    assert visible == visible_combat(other)
    assert len(visible['encounter']) == 2
    assert visible['encounter'][0]['hp'] == 0
    np.testing.assert_array_equal(features(visible, 1024), features(visible_combat(other), 1024))


def test_public_change_is_visible():
    raw = dict(player=dict(hp=80), deck={})
    other = deepcopy(raw)
    other['player']['hp'] = 79
    assert not np.array_equal(features(visible_combat(raw), 1024), features(visible_combat(other), 1024))
