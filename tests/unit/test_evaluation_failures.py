"""Failure boundaries preserve evidence and leave the UI out of running state."""
import csv
import json
import sys
from types import SimpleNamespace

import pytest

from apps import dashboard
from rl.evaluation import eval as cli


@pytest.mark.parametrize('failure', [TimeoutError, KeyboardInterrupt])
def test_cli_preserves_completed_rows_and_propagates_failure(tmp_path, monkeypatch, failure):
    output, table = tmp_path/'eval.json', tmp_path/'eval.csv'
    monkeypatch.setattr(sys, 'argv', ['eval', 'model.zip', '--output', str(output),
                                    '--output-csv', str(table), '--episodes', '2'])
    monkeypatch.setattr(cli, 'load_checkpoint', lambda *a, **kw: SimpleNamespace(set_random_seed=lambda seed: None))
    monkeypatch.setitem(cli.STRATEGIES, 'policy', lambda *a: SimpleNamespace(name='fixture'))
    monkeypatch.setattr(cli, 'make_scenario', lambda *a: None)

    def interrupted(*a, **kw):
        yield dict(episode=0, seed=10000, result='win', floor=55, act=4, steps=3, raw_return=1.)
        raise failure('budget exhausted')
    monkeypatch.setattr(cli, 'evaluate', interrupted)
    with pytest.raises(failure, match='budget exhausted'):
        cli.main()
    payload = json.loads(output.read_text(encoding='utf-8'))
    assert payload['state'] == 'error'
    assert payload['error']['type'] == failure.__name__
    assert payload['summary']['completed'] == 1
    assert payload['rows'][0]['seed'] == 10000
    with table.open(encoding='utf-8-sig', newline='') as stream:
        assert list(csv.DictReader(stream))[0]['result'] == 'win'


def test_dashboard_marks_strategy_initialization_failure():
    state = dashboard.DashboardState()
    state.eval_token = 'active'
    state.eval_status = {'state': 'running'}
    # Invalid native parameters fail before the first environment is created.
    state._evaluate('active', object(), 1, 0,
                    {'method': 'nosl_is_az_cpp', 'simulations': -1}, 1,
                    'model', 'model')
    assert state.eval_status['state'] == 'error'
    assert 'simulations' in state.eval_status['error']
    assert state.eval_status['active_workers'] == 0
