import numpy as np
import pytest

from tools.analysis.probe_upper_returns import annotate_returns, independent_reward


def test_complete_macro_returns_match_independent_reward_ledger():
    records = [dict(env=0, episode=0, terminal=False, reward=0., value=.2, bootstrap=.4, end=31),
               dict(env=0, episode=0, terminal=False, reward=.1, value=.4, bootstrap=.3, end=130),
               dict(env=0, episode=0, terminal=True, reward=1., value=.3, bootstrap=0., end=150)]
    result = annotate_returns(records, True)
    assert result['lambda1_max_error'] < 2e-6
    np.testing.assert_allclose([r['mc_return'] for r in records], [1.1, 1.1, 1.])
    # First macro in a 128-step collection cannot see the .1 and terminal bonus;
    # a 256-step collection can. The pending 31->130 macro belongs to its finish.
    assert records[0]['advantage_n128'] == pytest.approx(.2)
    assert records[0]['advantage_n256'] == pytest.approx(.2 + .95 * (.95 * .7))
    assert records[1]['advantage_n128'] == pytest.approx(.95 * .7)


def test_censored_prefix_is_not_monte_carlo():
    rows = [dict(terminal=False, bootstrap=None)]
    assert annotate_returns(rows, False) == dict(complete=False, macros=1)
    assert 'mc_return' not in rows[0]


def test_independent_terminal_progress_ledger():
    before = dict(act=2, floor=33)
    after = dict(act=3, floor=34, outcome='UNDECIDED')
    assert independent_reward(before, after, False, False) == pytest.approx(.202)
    assert independent_reward(before, after, False, True) == 0
    assert independent_reward(after, dict(after, outcome='PLAYER_VICTORY'), True, False) == 1
