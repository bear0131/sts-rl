import json
from types import SimpleNamespace
from unittest.mock import patch

from apps.training_monitor import Monitor, clean, merge_rows, allowed_client


def test_lan_access_is_explicit_and_loopback_stays_available():
    import ipaddress
    networks = [ipaddress.ip_network('192.168.20.0/24')]
    assert allowed_client('127.0.0.1', [])
    assert allowed_client('::1', [])
    assert allowed_client('192.168.20.15', networks)
    assert not allowed_client('192.168.21.15', networks)
    assert not allowed_client('8.8.8.8', networks)
    assert not allowed_client('192.168.20.15', [])


def test_resume_rows_replace_duplicate_steps_without_losing_history():
    before = [{'global_timesteps': 10, 'loss': 1}, {'global_timesteps': 20, 'loss': 2}]
    incoming = [{'global_timesteps': 20, 'loss': 3}, {'global_timesteps': 30, 'loss': 4}]
    assert merge_rows(before, incoming) == [before[0], incoming[0], incoming[1]]


def test_nonfinite_metrics_are_missing_not_zero():
    assert clean({'loss': float('nan'), 'values': [float('inf'), -0.1]}) == {
        'loss': None, 'values': [None, -0.1]}


def test_new_campfire_metrics_do_not_backfill_historical_rows():
    old = {'global_timesteps': 10}
    new = {'global_timesteps': 20, 'upper/campfire_rest_choice_rate': .25,
           'upper/campfire_rest_policy_probability': .3}
    rows = clean(merge_rows([old], [new]))
    assert rows == [old, new]
    assert 'upper/campfire_rest_choice_rate' not in rows[0]
    assert 'upper/campfire_dig_choice_rate' not in rows[1]


def test_connection_failure_preserves_cache_and_marks_stale(tmp_path):
    config = dict(run_id='example', iteration_steps=10, ports=[2200], ssh_key='key',
                  root='/repo', host='root@example.com')
    output = tmp_path / 'state.json'
    output.write_text(json.dumps(dict(run_id='example', rows=[{'global_timesteps': 10}],
                                     ranks=[dict(rank=0, checked_at=100, alive=True,
                                                 cursor=dict(offset=20, inode=1))])))
    m = Monitor(config, output, 30)
    with patch('apps.training_monitor.subprocess.run', side_effect=OSError('unreachable')):
        m.poll()
    snapshot = json.loads(m.snapshot())
    assert snapshot['rows'] == [{'global_timesteps': 10}]
    assert snapshot['ranks'][0]['checked_at'] == 100
    assert snapshot['ranks'][0]['cursor']['offset'] == 20
    assert snapshot['ranks'][0]['connection_error'] == 'unreachable'
    assert json.loads(output.read_text()) == snapshot
    other = Monitor(dict(config, run_id='another'), output, 30)
    assert other.state['rows'] == []


def test_ssh_stdin_uses_literal_lf_on_windows(tmp_path):
    config = dict(run_id='example', iteration_steps=10, ports=[2200], ssh_key='key',
                  root='/repo with space', host='root@example.com')
    monitor = Monitor(config, tmp_path / 'state.json', 30)
    with patch('apps.training_monitor.subprocess.run', return_value=SimpleNamespace(
            returncode=0, stdout=b'{"rank": 0, "alive": true}', stderr=b'')) as run:
        assert monitor.read_rank(0)['alive']
    script = run.call_args.kwargs['input']
    assert isinstance(script, bytes) and b'\r' not in script
    assert script.startswith(b"cd '/repo with space'\n")
    assert script.endswith(b'\nPY_MONITOR\n')


def test_comparison_keeps_same_steps_separate_and_survives_restart(tmp_path):
    old = tmp_path / 'old.json'
    old.write_text(json.dumps(dict(run_id='old', iteration_steps=10,
                                  rows=[dict(global_timesteps=20, loss=1.)])))
    config = dict(run_id='new', iteration_steps=10, ports=[2200], branch_step=20,
                  comparisons=[dict(run_id='old', history_file=str(old), branch_step=10)])
    output = tmp_path / 'new.json'
    output.write_text(json.dumps(dict(run_id='new', rows=[dict(global_timesteps=20, loss=2.)])))
    m = Monitor(config, output, 30)
    snapshot = json.loads(m.snapshot())
    assert snapshot['rows'][0]['loss'] == 2.
    assert snapshot['comparisons'][0]['rows'][0]['loss'] == 1.
    assert snapshot['branch_step'] == 20
    assert snapshot['comparisons'][0]['branch_step'] == 10
    assert json.loads(old.read_text())['rows'][0]['loss'] == 1.


def test_rollout_length_change_keeps_common_branch_round_after_reload(tmp_path):
    old = tmp_path / 'old.json'
    old.write_text(json.dumps(dict(run_id='old', iteration_steps=786432, rows=[])))
    config = dict(run_id='longer', iteration_steps=1179648, branch_step=169082880,
                  branch_round=215, ports=[2200], comparisons=[dict(run_id='old',
                  history_file=str(old), branch_step=169082880)])
    output = tmp_path / 'new.json'
    output.write_text(json.dumps(dict(run_id='longer', iteration_steps=786432,
                                      rows=[dict(global_timesteps=170262528)])))
    state = json.loads(Monitor(config, output, 30).snapshot())
    assert state['iteration_steps'] == 1179648 and state['branch_round'] == 215
    assert state['comparisons'][0]['branch_round'] == 215
    assert state['comparisons'][0]['iteration_steps'] == 786432
    assert state['rows'][0]['global_timesteps'] == 170262528  # Never rewrite samples.


def test_complete_game_rollouts_preserve_variable_actions_and_explicit_rounds(tmp_path):
    config = dict(run_id='fresh960', iteration_steps=None, rollout_mode='complete_games',
                  branch_round=0, branch_step=0, ports=[2200])
    output = tmp_path/'state.json'
    rows = [dict(global_timesteps=12003, rollout_round=1), dict(global_timesteps=26001, rollout_round=2)]
    output.write_text(json.dumps(dict(run_id='fresh960', rows=rows)))
    state = json.loads(Monitor(config, output, 30).snapshot())
    assert state['iteration_steps'] is None and state['rollout_mode'] == 'complete_games'
    assert state['rows'] == rows


def test_dashboard_uses_complete_cohorts_and_epoch_summaries(tmp_path):
    """Render conflicting legacy/complete metrics without a browser dependency."""
    from pathlib import Path
    import subprocess

    page = Path('apps/dashboard_static/training.html').read_text(encoding='utf-8')
    assert '<summary>历史数据说明</summary>' in page
    assert '<details open' not in page
    script = page.split('<script>', 1)[1].split('</script>', 1)[0]
    harness = r'''
const assert = require('node:assert/strict');
const vm = require('node:vm');
const nodes = new Map();
function element() {
 return {textContent:'',style:{},clientWidth:600,value:'0',children:[],
  setAttribute(){},append(...values){this.children.push(...values)},
  replaceChildren(){this.children=[];this.textContent=''}};
}
const document = {
 getElementById(id){if(!nodes.has(id)){const e=element();e.previousElementSibling=element();nodes.set(id,e)}return nodes.get(id)},
 createElement:element,createElementNS:element
};
document.getElementById('axis').value='absolute';
const context = vm.createContext({document,assert,ResizeObserver:class{observe(){}},
 fetch:()=>new Promise(()=>{}),setInterval(){}});
vm.runInContext(SCRIPT,context);
vm.runInContext(`
const old={global_timesteps:1,'game/win_rate':1,'game/floor_mean':99,
 'upper/policy_loss':-100,'upper/value_loss':100,'upper/kl':1};
const first={global_timesteps:10,rollout_round:1,games_per_rollout_global:240,
 'episode_rollout/completed_games_global':240,'episode_rollout/win_rate':.25,'episode_rollout/floor_mean':10};
const last={...old,global_timesteps:20,rollout_round:2,games_per_rollout_global:960,
 'episode_rollout/completed_games_global':960,'episode_rollout/win_rate':.75,'episode_rollout/floor_mean':40,
 'upper/epochs_completed':1,'upper/kl_stopped':1,
 'upper/policy_loss_epoch_1':.02,'upper/policy_loss_epoch_2':-.001,
 'upper/value_loss_epoch_1':.3,'upper/value_loss_epoch_2':.1,
 'upper/kl_epoch_1':.003,'upper/kl_epoch_2':.005};
state={run_id:'current',rollout_mode:'complete_games',branch_step:0,branch_round:0,
 updated_at:Date.now()/1000,rows:[first,last],ranks:[],
 comparisons:[{run_id:'legacy',iteration_steps:1,branch_step:0,rows:[old]}]};
const before=JSON.stringify(state);
render();
assert.equal($('win').textContent,'75.00%');
assert.equal($('floor').textContent,'40');
assert.equal($('winavg').textContent,'最近2个完整轮次 · 1,200局 · 65.00%');
assert.equal($('chart0').previousElementSibling.children.length,2);
assert.equal($('chart0').previousElementSibling.children[0].textContent.includes('legacy'),true);
assert.equal(displayRow(last)['display/upper_policy_loss'],-.001);
assert.equal(displayRow(last)['display/upper_value_loss'],.1);
assert.equal(displayRow(last)['display/upper_kl'],.005);
assert.equal(lastEpochMetric({'upper/kl_epoch_9':9,'upper/kl_epoch_10':10},'kl'),10);
assert.equal(displayRow(old)['display/upper_policy_loss'],undefined);
assert.equal(displayRow(old)['episode_rollout/win_rate'],undefined);
assert.equal(completeCohort({...last,'episode_rollout/completed_games_global':900}),null);
assert.equal(cohortSummary([old]),null);
assert.equal(displayRow(old,state)['display/game_win_rate'],undefined);
assert.equal(displayRow(old,state.comparisons[0])['display/game_win_rate'],1);
assert.equal(displayRow({...last,'episode_rollout/completed_games_global':900},state)['display/game_win_rate'],undefined);
assert.ok($('chart0').previousElementSibling.children[0].textContent.includes('最近400个完成局'));
assert.ok($('comparison-status').children.some(n=>n.textContent.includes('legacy')&&n.textContent.includes('最近400个完成局')));
assert.equal(JSON.stringify(state),before); // Display must not rewrite raw/history data.
assert.ok(!specs.flatMap(s=>s[1]).some(k=>['game/win_rate','game/floor_mean','upper/policy_loss','upper/value_loss','upper/kl'].includes(k)));
const reference={...first,global_timesteps:8,rollout_round:1};
state.comparisons.push({run_id:'lambda-reference',label:'lambda-reference',rollout_mode:'complete_games',branch_step:0,branch_round:0,rows:[reference]});
render();
assert.equal($('chart0').previousElementSibling.children.length,3);
assert.ok($('chart0').previousElementSibling.children.some(n=>n.textContent.includes('lambda-reference')));
assert.equal(datasets().find(s=>s.label==='lambda-reference').x(reference),1); // Explicit rounds, different action counts.
state.comparisons.pop();
state.rows=[old];state.comparisons=[];render();
assert.equal($('win').textContent,'—');assert.equal($('floor').textContent,'—');
assert.equal($('winavg').textContent,'暂无完整轮次统计');
assert.equal($('chart0').previousElementSibling.textContent,'暂无同口径数据');
`,context);
'''.replace('SCRIPT', json.dumps(script))
    path = tmp_path / 'dashboard.cjs'
    path.write_text(harness, encoding='utf-8')
    subprocess.run(['node', str(path)], check=True, capture_output=True, timeout=15)
