"""Local training charts and user worklog messages; no model loading."""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import ipaddress
import math
import os
import re
import secrets
import tempfile
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
import shlex
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


from apps.training_budget import budget_rows, merge_budget_records


REMOTE = r'''
import json, os, pathlib, time
root=pathlib.Path(ROOT); run=root/'runs'/RUN
pidfile=run/f'rank{RANK}.pid'
pid=int(pidfile.read_text()) if pidfile.exists() else None
proc=pathlib.Path(f'/proc/{pid}')
alive=False
if pid and (proc/'cmdline').exists():
    cmd=(proc/'cmdline').read_bytes().split(b'\0')
    alive=(b'rl.training.distributed_sb3' in cmd and RUN.encode() in cmd
           and (proc/'cwd').resolve().is_relative_to(root.resolve()))
log=run/f'rank{RANK}.log'; st=log.stat()
cursor=CURSOR
offset=cursor.get('offset',0) if cursor.get('inode')==st.st_ino else 0
if offset>st.st_size: offset=0
rows=[]
if RANK==0:
    with log.open('rb') as f:
        f.seek(offset); data=f.read(16*1024*1024)
    complete=data.rfind(b'\n')+1
    for line in data[:complete].splitlines():
        if not line.startswith(b'{'): continue
        try: row=json.loads(line)
        except (ValueError,UnicodeDecodeError): continue
        if 'global_timesteps' in row and 'rollout_sec' in row: rows.append(row)
    offset+=complete
budget_offset=cursor.get('budget_offset',0) if cursor.get('inode')==st.st_ino else 0
if budget_offset>st.st_size: budget_offset=0
budget_records=[]
with log.open('rb') as f:
    f.seek(budget_offset); budget_data=f.read(16*1024*1024)
budget_complete=budget_data.rfind(b'\n')+1
for line in budget_data[:budget_complete].splitlines():
    if not line.startswith(b'{'): continue
    try: row=json.loads(line)
    except (ValueError,UnicodeDecodeError): continue
    if 'global_timesteps' not in row or 'rollout_sec' not in row: continue
    stats={k:v for k,v in row.get('small_playout_stats_rank',{}).items() if '/budget_' in k}
    if stats or 'small_playout_train/budget_round' in row:
        budget_records.append(dict(rank=RANK,global_timesteps=row['global_timesteps'],
            rollout_round=row.get('rollout_round'),stats=stats))
budget_offset+=budget_complete
with log.open('rb') as f:
    f.seek(max(0,st.st_size-262144)); tail=f.read().decode('utf-8','replace').splitlines()
progress=None; launch=False; errors=[]
for line in tail:
    if line.startswith('=== NOSL launch '): errors=[];progress=None;launch=True
    if line.startswith('ROLLOUT_PROGRESS '):
        try: progress=json.loads(line.removeprefix('ROLLOUT_PROGRESS '))
        except ValueError: pass
    if any(x in line for x in ('Traceback','stack smashing detected','CUDA out of memory','Fatal Python error:')): errors.append(line[-300:])
steps=[]
for p in (run/'recovery'/f'rank{RANK}').glob('step-*'):
    if (p/'manifest.json').exists(): steps.append(int(p.name[5:]))
print(json.dumps(dict(rank=RANK,pid=pid,alive=alive,step=max(steps,default=0),
    progress=progress,log_mtime=st.st_mtime,errors=errors[-3:],rows=rows,budget_records=budget_records,
    cursor=dict(inode=st.st_ino,offset=offset,budget_offset=budget_offset),checked_at=time.time())))
'''


def clean(value):
    """JSON consumers must see missing nonfinite values, never fake zeroes."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clean(v) for v in value]
    return value


def merge_rows(existing, incoming):
    rows = {int(r['global_timesteps']): r for r in existing + incoming}
    return [rows[step] for step in sorted(rows)]


def allowed_client(address, networks):
    address = ipaddress.ip_address(address)
    return address.is_loopback or any(address in network for network in networks)


def monitor_target(value):
    """Validate the shared declarative identity before changing a live reader."""
    if not isinstance(value, dict):
        raise ValueError('活动监控声明缺失')
    required = ('root', 'run_id', 'label', 'rollout_mode', 'iteration_steps',
                'branch_step', 'branch_round', 'status', 'note')
    if any(key not in value for key in required):
        raise ValueError('活动监控声明缺少字段')
    if (not isinstance(value['root'], str) or not value['root'].startswith('/') or
        not isinstance(value['run_id'], str) or
        not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', value['run_id'])):
        raise ValueError('活动监控路径或run身份无效')
    if value['rollout_mode'] not in ('complete_games', 'fixed_steps'):
        raise ValueError('活动监控统计模式无效')
    if value['rollout_mode'] == 'fixed_steps' and (
        not isinstance(value['iteration_steps'], int) or value['iteration_steps'] <= 0):
        raise ValueError('固定步数模式缺少正整数iteration_steps')
    if value['status'] not in ('training', 'diagnostic', 'evaluation', 'paused', 'completed'):
        raise ValueError('活动监控阶段无效')
    if any(not isinstance(value[k], (int, float)) or not math.isfinite(value[k])
           or value[k] < 0 for k in ('branch_step', 'branch_round')):
        raise ValueError('活动监控分叉起点无效')
    if not isinstance(value['label'], str) or not isinstance(value['note'], str):
        raise ValueError('活动监控说明无效')
    result = {key: value[key] for key in required}
    predecessors = value.get('continuation_of', [])
    if not isinstance(predecessors, list):
        raise ValueError('续训历史声明必须是列表')
    seen = {(value['root'], value['run_id'])}
    for source in predecessors:
        if (not isinstance(source, dict) or
            not isinstance(source.get('root'), str) or not source['root'].startswith('/') or
            not isinstance(source.get('run_id'), str) or
            not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', source['run_id']) or
            type(source.get('through_step')) is not int or source['through_step'] < 0):
            raise ValueError('续训历史身份或截止步数无效')
        identity = (source['root'], source['run_id'])
        if identity in seen:
            raise ValueError('续训历史身份重复')
        seen.add(identity)
    if predecessors:
        result['continuation_of'] = [{k: source[k] for k in ('root', 'run_id', 'through_step')}
                                     for source in predecessors]
    return result


class WorklogFeed:
    """Read committed worklogs from one configured remote branch, never checkout."""
    def __init__(self, config, cache):
        self.config, self.cache = config, cache
        self.lock = threading.Lock()
        self.git_lock = threading.Lock()
        self.send_token = secrets.token_urlsafe(32)
        self.state = dict(enabled=bool(config), updated_at=None, error=None, entries=[])
        if config and cache.exists():
            try:
                saved = json.loads(cache.read_text(encoding='utf-8'))
                if saved.get('branch') == config['branch']:
                    self.state.update(saved)
            except (OSError, ValueError):
                pass

    def git(self, *args, input=None, extra_env=None):
        with self.git_lock:
            return self._git(*args, input=input, extra_env=extra_env)

    def _git(self, *args, input=None, extra_env=None):
        result = subprocess.run(['git', '-C', self.config['repository'], *args],
            capture_output=True, timeout=25, input=input,
            env=dict(os.environ, GIT_TERMINAL_PROMPT='0',
                     GIT_SSH_COMMAND='ssh -o BatchMode=yes -o ConnectTimeout=8',
                     **(extra_env or {})))
        if result.returncode:
            raise RuntimeError('Git同步失败（退出码 %s）' % result.returncode)
        return result.stdout.decode('utf-8', 'replace')

    def poll(self):
        if not self.config:
            return
        try:
            branch = self.config['branch']
            self.git('check-ref-format', 'refs/heads/' + branch)
            self.git('fetch', '--no-tags', 'origin',
                     'refs/heads/' + branch + ':refs/remotes/origin/' + branch)
            commit = self.git('rev-parse', 'refs/remotes/origin/' + branch).strip()
            paths = sorted(p for p in self.git('ls-tree', '-r', '--name-only',
                commit, '--', 'worklog/').splitlines()
                if re.fullmatch(r'worklog/\d{4}-\d{2}-\d{2}\.md', p))[-3:]
            entries = []
            for path in reversed(paths):
                content = self.git('show', commit + ':' + path)
                entries.append(dict(path=path, text=content[-300000:],
                                    truncated=len(content) > 300000))
            active_monitor, monitor_error = None, None
            registry = self.config.get('monitor_registry')
            if registry:
                try:
                    data = json.loads(self.git('show', commit + ':' + registry))
                    active_monitor = monitor_target(data.get('monitor'))
                except (ValueError, AttributeError, RuntimeError) as exc:
                    monitor_error = '活动任务声明无效，保留现有曲线'
            state = dict(active_monitor=active_monitor, monitor_error=monitor_error,
                enabled=True, branch=branch, commit=commit,
                updated_at=time.time(), attempted_at=time.time(), error=None, entries=entries)
        except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired):
            with self.lock:
                self.state.update(attempted_at=time.time(), error='远端日志同步失败，保留上次成功内容')
            return
        with self.lock:
            self.state = state
        self.cache.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.cache.with_suffix('.tmp')
        tmp.write_text(json.dumps(state, ensure_ascii=False), encoding='utf-8')
        os.replace(tmp, self.cache)

    def snapshot(self):
        with self.lock:
            return json.dumps(dict(self.state, can_send=bool(self.config and
                self.config.get('allow_messages')), send_token=self.send_token),
                ensure_ascii=False).encode('utf-8')


class WorklogWriter:
    """Append user text with optimistic Git commits; never alter a checkout."""
    def __init__(self, feed, receipts):
        self.feed, self.receipts = feed, receipts
        self.lock = threading.Lock()
        self.messages = {}
        if receipts.exists():
            self.messages = json.loads(receipts.read_text(encoding='utf-8'))

    def save(self):
        self.receipts.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.receipts.with_suffix('.tmp')
        tmp.write_text(json.dumps(self.messages, ensure_ascii=False), encoding='utf-8')
        os.replace(tmp, self.receipts)

    def publish(self, message_id, text):
        if not self.feed.config or not self.feed.config.get('allow_messages'):
            raise ValueError('此监控尚未启用用户发言')
        if not isinstance(text, str) or not text.strip() or len(text) > 20000:
            raise ValueError('请输入1至20000字的发言')
        if not isinstance(message_id, str) or str(uuid.UUID(message_id)) != message_id:
            raise ValueError('发送编号无效，请刷新页面')
        if not self.lock.acquire(blocking=False):
            raise RuntimeError('已有发言正在发送，请稍后重试')
        try:
            receipt = self.messages.get(message_id)
            if receipt and receipt['text'] != text:
                raise ValueError('同一发送编号的内容已改变，请重新编辑后发送')
            if receipt and receipt.get('status') == 'published':
                return {k: v for k, v in receipt.items() if k != 'text'}
            now = datetime.now(timezone(timedelta(hours=8)))
            if not receipt:
                receipt = dict(id='USER-' + message_id, text=text,
                    path='worklog/' + now.strftime('%Y-%m-%d') + '.md',
                    created_at=now.isoformat(timespec='seconds'), status='pending')
                self.messages[message_id] = receipt
            self.save()
            marker = '<!-- user-message:' + message_id + ' -->'
            branch = self.feed.config['branch']
            git = self.feed.git
            git('check-ref-format', 'refs/heads/' + branch)
            # Every retry starts from the current remote tip, preserving concurrent work.
            for attempt in range(3):
                git('fetch', '--no-tags', 'origin',
                    'refs/heads/' + branch + ':refs/remotes/origin/' + branch)
                base = git('rev-parse', 'refs/remotes/origin/' + branch).strip()
                exists = git('ls-tree', '--name-only', base, '--', receipt['path']).strip()
                previous = git('show', base + ':' + receipt['path']) if exists else ''
                if marker in previous:
                    receipt.update(status='published', commit=base)
                    self.save()
                    return {k: v for k, v in receipt.items() if k != 'text'}
                # Quote every user line so Markdown headings cannot impersonate log metadata.
                quoted = '\n'.join('> ' + line for line in text.split('\n'))
                addition = ('\n\n## 用户发言 · ' + receipt['id'] + '\n\n' + marker +
                    '\n- 时间：' + receipt['created_at'] +
                    '\n- 收件对象：当前远端训练管理会话。来源：用户在8773前端主动发送。' +
                    '\n- 请注意本条用户prompt，读取后在日志末尾引用本ID回复接收/处理状态。' +
                    '\n- 以下为用户原文；发送成功仅表示已推送，尚未确认对方读取或执行。\n\n' + quoted + '\n')
                blob = git('hash-object', '-w', '--stdin',
                           input=(previous + addition).encode('utf-8')).strip()
                with tempfile.TemporaryDirectory(prefix='worklog-index-',
                                                 dir=self.receipts.parent) as folder:
                    env = dict(GIT_INDEX_FILE=str(Path(folder) / 'index'))
                    git('read-tree', base, extra_env=env)
                    git('update-index', '--add', '--cacheinfo',
                        '100644,' + blob + ',' + receipt['path'], extra_env=env)
                    tree = git('write-tree', extra_env=env).strip()
                commit = git('commit-tree', tree, '-p', base,
                    input=('Record user prompt ' + receipt['id'] + '\n').encode()).strip()
                try:
                    git('push', 'origin', commit + ':refs/heads/' + branch)
                except (RuntimeError, subprocess.TimeoutExpired):
                    # A lost push response may still mean success; fetch/deduplicate next.
                    continue
                receipt.update(status='published', commit=commit)
                self.save()
                return {k: v for k, v in receipt.items() if k != 'text'}
            raise RuntimeError('推送未确认成功，请重试；相同编号不会重复写入')
        finally:
            self.lock.release()


class Monitor:
    def __init__(self, config, output, interval):
        config = dict(config)
        saved = json.loads(output.read_text(encoding='utf-8')) if output.exists() else {}
        if config.get('follow_active') and saved.get('active_target'):
            config.update(monitor_target(saved['active_target']))
        self.config, self.output, self.interval = config, output, interval
        self.poll_lock = threading.Lock()
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.state = dict(run_id=config['run_id'], iteration_steps=config['iteration_steps'],
                          interval=interval, rows=[], ranks=[], updated_at=None, polling=False)
        if saved.get('run_id') == config['run_id']:
            self.state.update(saved)
        self.state.update(interval=interval, polling=False, polling_ended=False, poll_error=None)
        self.state['label'] = config.get('label', config['run_id'])
        self.state['branch_step'] = config.get('branch_step', 0)
        self.state['iteration_steps'] = config['iteration_steps']
        self.state['rollout_mode'] = config.get('rollout_mode', 'fixed_steps')
        self.state['continuation_of'] = config.get('continuation_of', [])
        self.state['branch_round'] = config.get('branch_round',
            self.state['branch_step'] / (config['iteration_steps'] or 1))
        followed_histories = saved.get('comparisons', [])
        self.state['comparisons'] = []
        for reference in config.get('comparisons', []):
            saved = json.loads(Path(reference['history_file']).read_text(encoding='utf-8'))
            if saved['run_id'] != reference['run_id']:
                raise ValueError('comparison history belongs to a different run')
            self.state['comparisons'].append(dict(run_id=saved['run_id'],
                label=reference.get('label', saved['run_id']), rows=saved['rows'],
                iteration_steps=saved['iteration_steps'], rollout_mode=saved.get('rollout_mode', 'fixed_steps'), branch_step=reference.get('branch_step', 0),
                branch_round=reference.get('branch_round', saved.get('branch_round',
                    reference.get('branch_step', 0) / (saved['iteration_steps'] or 1)))))


        for history in followed_histories:
            if config.get('follow_active') and history.get('followed'):
                self.state['comparisons'].append(history)

    def save_locked(self):
        payload = json.dumps(clean(self.state), ensure_ascii=False, allow_nan=False)
        self.output.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.output.with_suffix('.tmp')
        tmp.write_text(payload, encoding='utf-8')
        os.replace(tmp, self.output)

    def follow_target(self, value, commit):
        if not self.config.get('follow_active'):
            return
        target = monitor_target(value)
        with self.poll_lock, self.lock:
            changed = (target['root'], target['run_id']) != (
                self.config['root'], self.config['run_id'])
            if changed:
                old = {k: self.state.get(k) for k in ('run_id', 'label', 'rows',
                    'iteration_steps', 'rollout_mode', 'branch_step', 'branch_round', 'continuation_of')}
                old.update(root=self.config['root'], followed=True)
                histories = self.state['comparisons']
                restored = next((h for h in histories if h.get('root') == target['root']
                                 and h['run_id'] == target['run_id']), {})
                self.state['comparisons'] = [h for h in histories if not (
                    h.get('root') in (old['root'], target['root']) and
                    (h.get('root'), h['run_id']) in (
                        (old['root'], old['run_id']), (target['root'], target['run_id'])))]
                self.state['comparisons'].append(old)
                self.state.update(rows=restored.get('rows', []), ranks=[],
                                  search_budget_records=[], search_budget_rows=[],
                                  updated_at=None, polling=False)
            self.config.update(target)
            self.state.update({k: target[k] for k in ('run_id', 'label',
                'iteration_steps', 'rollout_mode', 'branch_step', 'branch_round')})
            self.state.update(active_target=target, target_commit=commit, target_error=None)
            self.state['continuation_of'] = target.get('continuation_of', [])
            self.save_locked()

    def read_rank(self, rank):
        with self.lock:
            previous = next((s for s in self.state['ranks'] if s['rank'] == rank), {})
        c = self.config
        header = '\n'.join(f'{key}={value!r}' for key, value in dict(
            ROOT=c['root'], RUN=c['run_id'], RANK=rank, CURSOR=previous.get('cursor', {})).items())
        script = f"cd {shlex.quote(c['root'])}\n.venv/bin/python - <<'PY_MONITOR'\n{header}\n{REMOTE}\nPY_MONITOR\n"
        args = ['ssh', '-i', c['ssh_key'], '-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=yes',
                '-o', 'ConnectTimeout=8', '-o', 'ServerAliveInterval=10',
                '-p', str(c['ports'][rank]), c['host'], 'bash -se']
        try:
            # Text-mode stdin adds CRLF on Windows; bash must receive literal LF.
            result = subprocess.run(args, input=script.replace('\r\n', '\n').encode('utf-8'),
                                    capture_output=True, timeout=25)
            if result.returncode:
                detail = result.stderr.decode('utf-8', 'replace')[-250:]
                raise RuntimeError(f'SSH exit {result.returncode}: {detail}')
            value = json.loads(result.stdout)
            value['connection_error'] = None
            return value
        except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as exc:
            return dict(previous, rank=rank, connection_error=str(exc), attempted_at=time.time())

    def poll(self):
        with self.poll_lock:
            self.poll_ranks()

    def poll_ranks(self):
        with self.lock:
            self.state['polling'] = True
        with concurrent.futures.ThreadPoolExecutor(len(self.config['ports'])) as pool:
            ranks = list(pool.map(self.read_rank, range(len(self.config['ports']))))
        with self.lock:
            rows = [row for rank in ranks for row in rank.pop('rows', [])]
            self.state['rows'] = merge_rows(self.state['rows'], rows)
            records = [row for rank in ranks for row in rank.pop('budget_records', [])]
            self.state['search_budget_records'] = merge_budget_records(
                self.state.get('search_budget_records', []), records)
            self.state['search_budget_rows'] = budget_rows(
                self.state['search_budget_records'], len(self.config['ports']))
            self.state.update(ranks=ranks, updated_at=time.time(), polling=False, poll_error=None)
            self.save_locked()

    def watch(self, max_hours):
        deadline = time.monotonic() + max_hours * 3600 if max_hours else math.inf
        while not self.stop.is_set() and time.monotonic() < deadline:
            try:
                self.poll()
            except OSError:
                # Keep in-memory rows/cursors and retry after transient storage
                # failures; a live HTTP server must not hide a dead collector.
                with self.lock:
                    self.state.update(polling=False,
                        poll_error='训练数据缓存写入失败，正在自动重试')
            self.stop.wait(self.interval)
        with self.lock:
            self.state['polling_ended'] = True

    def snapshot(self):
        with self.lock:
            return json.dumps(clean(self.state), ensure_ascii=False, allow_nan=False).encode()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--port', type=int, default=8773)
    parser.add_argument('--listen', default='127.0.0.1',
                        help='Bind address; non-loopback requires explicit --allow-subnet.')
    parser.add_argument('--allow-subnet', action='append', default=[],
                        help='LAN CIDR allowed to read the dashboard; may be repeated.')
    parser.add_argument('--poll-seconds', type=float, default=30)
    parser.add_argument('--max-hours', type=float, default=48,
                        help='Polling duration; 0 continues until the local service stops.')
    args = parser.parse_args()
    try:
        listen = ipaddress.ip_address(args.listen)
        networks = [ipaddress.ip_network(value, strict=False) for value in args.allow_subnet]
    except ValueError as exc:
        parser.error(str(exc))
    if not listen.is_loopback and not networks:
        parser.error('non-loopback listener requires --allow-subnet')
    if not math.isfinite(args.poll_seconds) or not math.isfinite(args.max_hours) or args.poll_seconds < 10 or args.max_hours < 0:
        parser.error('finite poll-seconds >= 10 and max-hours >= 0 required')
    config = json.loads(args.config.read_text(encoding='utf-8'))
    if (config.get('rollout_mode') != 'complete_games' and (config.get('iteration_steps') or 0) <= 0) or not config['ports']:
        parser.error('positive iteration_steps and at least one rank required')
    monitor = Monitor(config, args.output, args.poll_seconds)
    worklog = WorklogFeed(config.get('worklog'), args.output.with_suffix('.worklog.json'))
    writer = WorklogWriter(worklog, args.output.with_suffix('.messages.json'))
    html = Path(__file__).with_name('dashboard_static') / 'training.html'

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if not allowed_client(self.client_address[0], networks):
                self.send_error(403)
                return
            if self.path == '/api/state':
                body, content = monitor.snapshot(), 'application/json; charset=utf-8'
            elif self.path == '/api/worklog':
                body, content = worklog.snapshot(), 'application/json; charset=utf-8'
            elif self.path == '/worklog':
                body = html.with_name('training_worklog.html').read_bytes()
                content = 'text/html; charset=utf-8'
            elif self.path in ('/', '/training.html'):
                body, content = html.read_bytes(), 'text/html; charset=utf-8'
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header('Content-Type', content)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            if not allowed_client(self.client_address[0], networks):
                self.send_error(403)
                return
            if self.path != '/api/messages':
                self.send_error(404)
                return
            origin = self.headers.get('Origin')
            expected_origin = 'http://' + self.headers.get('Host', '')
            if ((origin and origin != expected_origin) or
                self.headers.get('Content-Type') != 'application/json' or
                not secrets.compare_digest(self.headers.get('X-Monitor-Token', ''),
                                           worklog.send_token)):
                self.send_error(403)
                return
            self.connection.settimeout(15)
            try:
                size = int(self.headers.get('Content-Length', '0'))
                if not 0 < size <= 100000:
                    raise ValueError('发言过长或为空')
                payload = json.loads(self.rfile.read(size).decode('utf-8'))
                if not isinstance(payload, dict):
                    raise ValueError('发言格式无效')
                result = writer.publish(payload.get('id'), payload.get('text'))
                status = 200
            except (ValueError, UnicodeError):
                status, result = 400, dict(error='请输入有效发言（最多20000字）')
            except (OSError, RuntimeError, subprocess.TimeoutExpired):
                status, result = 503, dict(error='发送未确认成功，已保留草稿，请重试')
            body = json.dumps(result, ensure_ascii=False).encode('utf-8')
            self.send_response(status)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    # Charts remain read-only; explicit user messages only append to configured Git logs.
    server = ThreadingHTTPServer((str(listen), args.port), Handler)
    thread = threading.Thread(target=monitor.watch, args=(args.max_hours,), daemon=True)
    thread.start()
    def watch_worklog():
        while not monitor.stop.is_set():
            try:
                worklog.poll()
                if config.get('follow_active'):
                    with worklog.lock:
                        target = worklog.state.get('active_monitor')
                        commit = worklog.state.get('commit')
                        error = worklog.state.get('monitor_error') or worklog.state.get('error')
                    if target and not error:
                        monitor.follow_target(target, commit)
                    else:
                        with monitor.lock:
                            monitor.state['target_error'] = error or '等待活动任务声明'
            except Exception:
                # File-system failures must not silently kill the log collector.
                with worklog.lock:
                    worklog.state['error'] = '日志缓存写入失败，请检查本地磁盘'
            monitor.stop.wait(max(60, args.poll_seconds))
    threading.Thread(target=watch_worklog, daemon=True).start()
    print(f'Training monitor listening on {listen}:{args.port}', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        monitor.stop.set()
        server.server_close()


if __name__ == '__main__':
    main()
