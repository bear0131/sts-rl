import numpy as np
import pytest

from env.core import action_space as A
from env.search.upper_public import SearchBudget, supported_actions, backup


def ids(*segments):
    return [A.seg_base(s) + i for s, i in segments]


@pytest.mark.parametrize('phase,options,room,relics,actions,expected', [
    ('shop', {}, 'SHOP', [], [('SHOP_CARD', 0), ('SHOP_RELIC', 0), ('SHOP_REMOVE', 0), ('PROCEED', 0)], [0, 2, 3]),
    ('shop', {}, 'SHOP', ['The Courier'], [('SHOP_CARD', 0), ('SHOP_POTION', 0), ('SHOP_REMOVE', 0)], [2]),
    ('combat_reward', {'card_count': 1}, 'BOSS', [], [('REWARD_GOLD', 0), ('PROCEED', 0), ('REWARD_CARD', 0)], [0, 2]),
    ('combat_reward', {'card_count': 2}, 'MONSTER', [], [('REWARD_CARD', 0), ('REWARD_SKIP', 0), ('PROCEED', 0)], [2]),
    ('event', {'event': 'The Cleric'}, 'EVENT', [], [('EVENT_OPT', 0), ('EVENT_OPT', 1), ('EVENT_OPT', 2)], [0, 1, 2]),
    ('event', {'event': 'Shining Light'}, 'EVENT', [], [('EVENT_OPT', 0), ('EVENT_OPT', 1)], []),
    ('rest', {}, 'REST', ['Dream Catcher'], [('CAMPFIRE', 0), ('CAMPFIRE', 1), ('CAMPFIRE', 5)], [1]),
    ('card_select', {'select': {'context': 'TRANSFORM'}}, 'SHOP', [], [('PICK_CARD', 0)], []),
    ('card_select', {'select': {'context': 'REMOVE'}}, 'EVENT', [], [('PICK_CARD', 0)], []),
    ('map', {}, 'MONSTER', [], [('MAP_NODE', 0), ('MAP_NODE', 1)], []),
])
def test_public_action_boundary(phase, options, room, relics, actions, expected):
    legal = ids(*actions)
    state = dict(phase=phase, options=options, global_=None)
    state['global'] = dict(relics=relics)
    assert supported_actions(state, legal, room=room) == [legal[i] for i in expected]


def test_exact_backup_and_terminal_never_bootstraps():
    tree = dict(reason='ok', nodes=[
        dict(action=-1, reward=0., terminal=False, children=[1, 2]),
        dict(action=11, reward=.1, terminal=False, children=[3, 4]),
        dict(action=12, reward=1., terminal=True, children=[]),
        dict(action=21, reward=.2, terminal=False, children=[]),
        dict(action=22, reward=.3, terminal=False, children=[]),
    ])
    actions, values = backup(tree, {3: .4, 4: .2}, gamma=.9)
    np.testing.assert_array_equal(actions, [11, 12])
    np.testing.assert_allclose(values, [.1 + .9 * (.2 + .9 * .4), 1.])
    with pytest.raises(ValueError, match='exactly'):
        backup(tree, {2: 99., 3: .4, 4: .2})


@pytest.mark.parametrize('kwargs', [dict(depth=0), dict(transitions=-1), dict(seconds=float('nan'))])
def test_budget_rejects_unbounded_values(kwargs):
    with pytest.raises(ValueError):
        SearchBudget(**kwargs)
