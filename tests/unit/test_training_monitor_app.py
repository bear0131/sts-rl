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


def test_unbounded_polling_keeps_running_until_stopped(tmp_path):
    monitor = Monitor(dict(run_id='continuous', iteration_steps=1, ports=[2200]),
                      tmp_path/'state.json', 30)
    polls = []
    def poll():
        polls.append(1)
        if len(polls) == 2:
            monitor.stop.set()
    with patch.object(monitor, 'poll', side_effect=poll), patch.object(monitor.stop, 'wait'):
        monitor.watch(0)
    assert len(polls) == 2
    assert monitor.state['polling_ended']


def test_cache_write_failure_keeps_polling_and_recovers_without_losing_rows(tmp_path):
    import os
    output = tmp_path/'state.json'
    monitor = Monitor(dict(run_id='continuous', iteration_steps=1, ports=[2200]), output, 30)
    replace = os.replace
    writes, snapshots = [], []

    def fail_once(source, destination):
        writes.append(1)
        if len(writes) == 1:
            raise OSError(28, 'No space left on device')
        replace(source, destination)

    def observe(_timeout):
        snapshots.append(json.loads(monitor.snapshot()))
        if len(snapshots) == 2:
            monitor.stop.set()

    ranks = [dict(rank=0, alive=True, cursor=dict(offset=i),
                  rows=[dict(global_timesteps=i)]) for i in (10, 20)]
    with patch.object(monitor, 'read_rank', side_effect=ranks), \
            patch('apps.training_monitor.os.replace', side_effect=fail_once), \
            patch.object(monitor.stop, 'wait', side_effect=observe):
        monitor.watch(0)
    assert len(writes) == len(snapshots) == 2
    assert snapshots[0]['poll_error'] and not snapshots[0]['polling']
    assert snapshots[0]['rows'] == [dict(global_timesteps=10)]
    assert snapshots[1]['poll_error'] is None
    assert snapshots[1]['rows'] == [dict(global_timesteps=10), dict(global_timesteps=20)]
    assert snapshots[1]['ranks'][0]['cursor'] == dict(offset=20)
    assert json.loads(output.read_text()) == snapshots[1]


def test_dashboard_uses_complete_cohorts_and_epoch_summaries(tmp_path):
    """Render conflicting legacy/complete metrics without a browser dependency."""
    from pathlib import Path
    import subprocess

    page = Path('apps/dashboard_static/training.html').read_text(encoding='utf-8')
    assert '<summary>历史数据说明</summary>' in page
    assert 'upper/reward_relic_pickup_rate' in page and 'upper/reward_relic_policy_probability' in page
    assert '蓝钥匙换掉的那一件记为未领取' in page
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
 'upper/policy_loss_pre_update':.000012,
 'upper/value_mc_mse_pre_update':.42,'upper/value_mc_mse_post_update':.05,
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
assert.equal(displayRow(last)['display/upper_policy_loss'],.000012);
assert.equal(displayRow(last)['display/upper_value_loss'],.42);
assert.equal(displayRow(old)['display/upper_value_loss'],undefined);
assert.equal(displayRow({'upper/value_loss_epoch_1':.3,'upper/value_loss_epoch_4':.06})['display/upper_value_loss'],undefined);
assert.equal(displayRow({...last,'upper/value_loss_epoch_3':.08,'upper/value_loss_epoch_4':.06})['display/upper_value_loss'],.42);
assert.equal(displayRow(last)['display/upper_kl'],.005);
assert.equal(lastEpochMetric({'upper/kl_epoch_9':9,'upper/kl_epoch_10':10},'kl'),10);
assert.equal(displayRow(old)['display/upper_policy_loss'],undefined);
assert.equal(displayRow(old)['episode_rollout/win_rate'],undefined);
assert.equal(completeCohort({...last,'episode_rollout/completed_games_global':900}),null);
assert.equal(completeCohort({...last,'episode_rollout/completed_games_global':900,'episode_rollout/censored_games_global':60}).games,900);
assert.equal(completeCohort({...last,'episode_rollout/completed_games_global':900,'episode_rollout/censored_games_global':60}).win,.75);
assert.equal(completeCohort({...last,'episode_rollout/completed_games_global':900,'episode_rollout/censored_games_global':59}),null);
assert.equal(cohortSummary([old]),null);
assert.equal(displayRow(old,state)['display/game_win_rate'],undefined);
assert.equal(displayRow(old,state.comparisons[0])['display/game_win_rate'],1);
assert.equal(displayRow({...last,'episode_rollout/completed_games_global':900},state)['display/game_win_rate'],undefined);
assert.ok($('chart0').previousElementSibling.children[0].textContent.includes('最近400个完成局'));
assert.ok($('comparison-status').children.some(n=>n.textContent.includes('legacy')&&n.textContent.includes('最近400个完成局')));
assert.equal(JSON.stringify(state),before); // Display must not rewrite raw/history data.
state.poll_error='训练数据缓存写入失败，正在自动重试';render();
assert.ok($('connection').textContent.includes(state.poll_error));
assert.equal($('connection').className,'bad');
state.poll_error=null;render();assert.equal($('connection').className,'');
assert.ok(!specs.flatMap(s=>s[1]).some(k=>['game/win_rate','game/floor_mean','upper/policy_loss','upper/value_loss','upper/kl'].includes(k)));
const reference={...first,global_timesteps:8,rollout_round:1};
state.comparisons.push({run_id:'lambda-reference',label:'lambda-reference',rollout_mode:'complete_games',branch_step:0,branch_round:0,rows:[reference]});
render();
assert.equal($('chart0').previousElementSibling.children.length,3);
assert.ok($('chart0').previousElementSibling.children.some(n=>n.textContent.includes('lambda-reference')));
assert.equal(datasets().find(s=>s.label==='lambda-reference').x(reference),1); // Explicit rounds, different action counts.
state.comparisons.pop();
state.label='COMBAT37';
state.rows=[last];
state.continuation_of=[{root:'/old',run_id:'predecessor',through_step:10}];
state.comparisons.push({root:'/old',run_id:'predecessor',rollout_mode:'complete_games',rows:[first,{...last,global_timesteps:30}]});
const raw=JSON.stringify(state);
render();
assert.equal($('run').textContent,'COMBAT37');
assert.equal(datasets().length,2);
assert.equal(datasets().at(-1).rows.length,2);
assert.equal(datasets().at(-1).x(first),1);
assert.equal(datasets().at(-1).x(last),2);
assert.equal($('winavg').textContent,'最近2个完整轮次 · 1,200局 · 65.00%');
assert.equal(JSON.stringify(state),raw);
assert.equal(displayRow({'upper/policy_loss_epoch_1':.2})['display/upper_policy_loss'],undefined);
state.continuation_of=[];
state.rows=[old];state.comparisons=[];render();
assert.equal($('win').textContent,'—');assert.equal($('floor').textContent,'—');
assert.equal($('winavg').textContent,'暂无完整轮次统计');
assert.equal($('chart0').previousElementSibling.textContent,'暂无同口径数据');
state.rollout_mode='fixed_steps';state.iteration_steps=1;render();
assert.equal($('win').textContent,'100.00%');assert.equal($('floor').textContent,'99');
assert.equal($('win-label').textContent,'最近400个完成局胜率');
assert.ok($('winavg').textContent.includes('连续补局滚动窗口'));
`,context);
'''.replace('SCRIPT', json.dumps(script))
    path = tmp_path / 'dashboard.cjs'
    path.write_text(harness, encoding='utf-8')
    subprocess.run(['node', str(path)], check=True, capture_output=True, timeout=15)


def test_worklog_tracks_remote_commits_without_checkout_and_keeps_cache_on_failure(tmp_path):
    import subprocess
    from apps.training_monitor import WorklogFeed

    source, reader = tmp_path / 'source', tmp_path / 'reader'
    def git(path, *args):
        return subprocess.run(['git', '-C', str(path), *args], check=True,
                              capture_output=True).stdout.decode().strip()
    source.mkdir()
    git(source, 'init', '-b', 'codex/log')
    git(source, 'config', 'user.email', 'test@example.invalid')
    git(source, 'config', 'user.name', 'test')
    (source / 'worklog').mkdir()
    log = source / 'worklog/2026-11-03.md'
    log.write_text('## first\n<unsafe> & 中文', encoding='utf-8')
    git(source, 'add', '.')
    git(source, 'commit', '-m', 'first')
    subprocess.run(['git', 'clone', str(source), str(reader)], check=True, capture_output=True)
    head = git(reader, 'rev-parse', 'HEAD')
    config = dict(repository=str(reader), branch='codex/log')
    cache = tmp_path / 'worklog.json'
    feed = WorklogFeed(config, cache)
    feed.poll()
    first = json.loads(feed.snapshot())
    assert first['error'] is None
    assert first['entries'][0]['text'] == '## first\n<unsafe> & 中文'
    log.write_text('## second\nnew decision', encoding='utf-8')
    git(source, 'add', '.')
    git(source, 'commit', '-m', 'second')
    feed.poll()
    second = json.loads(feed.snapshot())
    assert second['commit'] != first['commit']
    assert second['entries'][0]['text'] == '## second\nnew decision'
    assert git(reader, 'rev-parse', 'HEAD') == head
    git(reader, 'remote', 'set-url', 'origin', str(tmp_path / 'missing'))
    feed.poll()
    failed = json.loads(feed.snapshot())
    assert failed['error']
    assert failed['entries'] == second['entries']
    assert failed['updated_at'] == second['updated_at']
    restarted = WorklogFeed(config, cache)
    assert json.loads(restarted.snapshot())['entries'] == second['entries']
    assert json.loads(WorklogFeed(None, tmp_path / 'disabled').snapshot())['enabled'] is False


import pytest


@pytest.fixture
def message_repositories(tmp_path):
    import subprocess
    from apps.training_monitor import WorklogFeed, WorklogWriter
    origin, source, reader = (tmp_path / name for name in ('origin', 'source', 'reader'))
    def git(path, *args):
        return subprocess.run(['git', '-C', str(path), *args], check=True,
                              capture_output=True).stdout.decode('utf-8').strip()
    origin.mkdir()
    git(origin, 'init', '--bare')
    source.mkdir()
    git(source, 'init', '-b', 'codex/log')
    git(source, 'config', 'user.email', 'test@example.invalid')
    git(source, 'config', 'user.name', 'test')
    (source / 'tracked').write_text('original', encoding='utf-8')
    git(source, 'add', '.')
    git(source, 'commit', '-m', 'initial')
    git(source, 'remote', 'add', 'origin', str(origin))
    git(source, 'push', 'origin', 'codex/log')
    subprocess.run(['git', 'clone', '-b', 'codex/log', str(origin), str(reader)],
                   check=True, capture_output=True)
    git(reader, 'config', 'user.email', 'test@example.invalid')
    git(reader, 'config', 'user.name', 'test')
    feed = WorklogFeed(dict(repository=str(reader), branch='codex/log',
                           allow_messages=True), tmp_path / 'cache.json')
    writer = WorklogWriter(feed, tmp_path / 'receipts.json')
    return SimpleNamespace(origin=origin, source=source, reader=reader,
                           feed=feed, writer=writer, git=git)


def test_user_message_preserves_concurrent_commits_checkout_and_staging(message_repositories):
    import uuid
    from datetime import datetime, timedelta, timezone
    r = message_repositories
    git = r.git
    original_head = git(r.reader, 'rev-parse', 'HEAD')
    (r.reader / 'tracked').write_text('user staged change', encoding='utf-8')
    git(r.reader, 'add', 'tracked')
    staged = git(r.reader, 'diff', '--cached')
    original_git = r.feed.git
    raced = False
    path = 'worklog/' + datetime.now(timezone(timedelta(hours=8))).strftime('%Y-%m-%d') + '.md'
    def concurrent_git(*args, **kwargs):
        nonlocal raced
        if args[0] == 'push' and not raced:
            raced = True
            (r.source / 'worklog').mkdir()
            (r.source / path).write_text('## Other session\nkeep this decision\n', encoding='utf-8')
            (r.source / 'other-code').write_text('keep code too', encoding='utf-8')
            git(r.source, 'add', '.')
            git(r.source, 'commit', '-m', 'concurrent work')
            git(r.source, 'push', 'origin', 'codex/log')
        return original_git(*args, **kwargs)
    with patch.object(r.feed, 'git', side_effect=concurrent_git):
        receipt = r.writer.publish(str(uuid.uuid4()), '中文 prompt\n## literal heading\n<script>not executable</script>')
    text = git(r.origin, 'show', 'refs/heads/codex/log:' + receipt['path'])
    assert receipt['status'] == 'published'
    assert 'keep this decision' in text
    assert '> 中文 prompt\n> ## literal heading\n> <script>not executable</script>' in text
    assert git(r.origin, 'show', 'refs/heads/codex/log:other-code') == 'keep code too'
    assert git(r.reader, 'rev-parse', 'HEAD') == original_head
    assert git(r.reader, 'diff', '--cached') == staged
    assert git(r.origin, 'diff', '--name-only', receipt['commit'] + '^', receipt['commit']) == receipt['path']


def test_lost_push_response_retries_once_and_restart_does_not_duplicate(message_repositories):
    import uuid
    import subprocess
    from apps.training_monitor import WorklogWriter
    r = message_repositories
    original_git = r.feed.git
    lost = False
    def lost_response(*args, **kwargs):
        nonlocal lost
        result = original_git(*args, **kwargs)
        if args[0] == 'push' and not lost:
            lost = True
            raise subprocess.TimeoutExpired('git', 25)
        return result
    message_id = str(uuid.uuid4())
    with patch.object(r.feed, 'git', side_effect=lost_response):
        receipt = r.writer.publish(message_id, 'Only once')
    restarted = WorklogWriter(r.feed, r.writer.receipts)
    assert restarted.publish(message_id, 'Only once') == receipt
    text = r.git(r.origin, 'show', 'refs/heads/codex/log:' + receipt['path'])
    assert text.count('<!-- user-message:' + message_id + ' -->') == 1
    with pytest.raises(ValueError):
        restarted.publish(message_id, 'Changed content')


def test_failed_message_is_not_published_and_can_be_retried(message_repositories):
    import uuid
    r = message_repositories
    message_id = str(uuid.uuid4())
    original = r.git(r.reader, 'remote', 'get-url', 'origin')
    r.git(r.reader, 'remote', 'set-url', 'origin', str(r.reader / 'missing'))
    with pytest.raises(RuntimeError):
        r.writer.publish(message_id, 'retry me')
    assert r.writer.messages[message_id]['status'] == 'pending'
    r.git(r.reader, 'remote', 'set-url', 'origin', original)
    assert r.writer.publish(message_id, 'retry me')['status'] == 'published'
    r.feed.config['allow_messages'] = False
    with pytest.raises(ValueError):
        r.writer.publish(str(uuid.uuid4()), 'disabled')


@pytest.fixture
def active_target():
    return dict(root='/original', run_id='original', label='Original',
        rollout_mode='complete_games', iteration_steps=None,
        branch_step=0, branch_round=0, status='training', note='current task')


def test_active_run_switches_without_mixing_rows_and_survives_restart(tmp_path, active_target):
    config = dict(active_target, ports=[2200], follow_active=True)
    history = tmp_path / 'legacy.json'
    history.write_text(json.dumps(dict(run_id='legacy', iteration_steps=10, rows=[])))
    config['comparisons'] = [dict(run_id='legacy', history_file=str(history))]
    output = tmp_path / 'state.json'
    m = Monitor(config, output, 30)
    m.state['rows'] = [dict(global_timesteps=10, rollout_round=1, loss=1)]
    m.state['ranks'] = [dict(rank=0, cursor=dict(offset=999, inode=1))]
    next_run = dict(active_target, root='/candidate', run_id='candidate', label='Candidate',
                    branch_step=10, branch_round=1)
    m.follow_target(next_run, 'commit1')
    assert m.state['rows'] == [] and m.state['ranks'] == []
    assert m.state['comparisons'][-1]['rows'][0]['loss'] == 1
    with patch.object(m, 'read_rank', return_value=dict(rank=0, rows=[
            dict(global_timesteps=10, rollout_round=2, loss=2)], connection_error=None)):
        m.poll()
    restarted = Monitor(config, output, 30)
    assert restarted.config['run_id'] == 'candidate'
    assert restarted.state['rows'][0]['loss'] == 2
    assert restarted.state['comparisons'][-1]['run_id'] == 'original'
    restarted.follow_target(active_target, 'commit2')
    assert restarted.state['rows'][0]['loss'] == 1
    assert restarted.state['comparisons'][-1]['rows'][0]['loss'] == 2
    restarted.follow_target(dict(active_target, status='diagnostic', note='frozen rollout'), 'commit3')
    assert restarted.state['active_target']['status'] == 'diagnostic'
    assert restarted.state['rows'][0]['loss'] == 1
    assert len(restarted.state['comparisons']) == 2
    with pytest.raises(ValueError):
        restarted.follow_target(dict(active_target, rollout_mode='fixed_steps'), 'invalid')
    assert restarted.config['run_id'] == 'original'


def test_continuation_metadata_survives_restart_without_merging_raw_cursors(tmp_path, active_target):
    from apps.training_monitor import monitor_target
    config = dict(active_target, ports=[2200], follow_active=True)
    output = tmp_path / 'state.json'
    m = Monitor(config, output, 30)
    m.state['rows'] = [dict(global_timesteps=10, rollout_round=1)]
    source = dict(root='/original', run_id='original', through_step=10)
    target = dict(active_target, root='/next', run_id='next', continuation_of=[source])
    m.follow_target(target, 'next')
    m.state['ranks'] = [dict(rank=0, cursor=dict(offset=123))]
    m.save_locked()
    restored = Monitor(config, output, 30)
    assert restored.state['continuation_of'] == [source]
    assert restored.state['rows'] == []
    assert restored.state['comparisons'][0]['rows'][0]['global_timesteps'] == 10
    assert restored.state['ranks'][0]['cursor']['offset'] == 123
    for sources in ([source, source], [dict(source, through_step=-1)],
                    [dict(source, through_step=True)], [dict(source, root='relative')], 'bad'):
        with pytest.raises(ValueError):
            monitor_target(dict(target, continuation_of=sources))
    with pytest.raises(ValueError):
        monitor_target(dict(active_target, continuation_of=[source]))
    restored.follow_target(active_target, 'back')
    assert restored.state['continuation_of'] == []


def test_registry_and_logs_share_commit_and_invalid_registry_keeps_logs_fresh(
        message_repositories, active_target):
    r = message_repositories
    r.feed.config['monitor_registry'] = 'registry.json'
    (r.source / 'registry.json').write_text(json.dumps(dict(monitor=active_target)))
    r.git(r.source, 'add', '.')
    r.git(r.source, 'commit', '-m', 'declare current run')
    r.git(r.source, 'push', 'origin', 'codex/log')
    r.feed.poll()
    first = json.loads(r.feed.snapshot())
    assert first['active_monitor'] == active_target
    assert first['monitor_error'] is None
    (r.source / 'registry.json').write_text('{invalid')
    r.git(r.source, 'add', '.')
    r.git(r.source, 'commit', '-m', 'bad registry')
    r.git(r.source, 'push', 'origin', 'codex/log')
    r.feed.poll()
    second = json.loads(r.feed.snapshot())
    assert second['commit'] != first['commit']
    assert second['monitor_error']
    assert second['active_monitor'] is None
    assert second['error'] is None
