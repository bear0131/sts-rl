"""Read-only local training charts; standard library only, no model loading."""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import ipaddress
import math
import os
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
    progress=progress,log_mtime=st.st_mtime,log_tail='\n'.join(tail[-120:])[-32000:],
    errors=errors[-3:],rows=rows,budget_records=budget_records,
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


class Monitor:
    def __init__(self, config, output, interval):
        if 'hosts' in config and (len(config['hosts']) != len(config['ports'])
                                  or any(not isinstance(host, str) or not host for host in config['hosts'])):
            raise ValueError('hosts must provide one SSH destination per rank port')
        self.config, self.output, self.interval = config, output, interval
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.state = dict(run_id=config['run_id'], iteration_steps=config['iteration_steps'],
                          interval=interval, rows=[], ranks=[], updated_at=None, polling=False)
        if output.exists():
            saved = json.loads(output.read_text(encoding='utf-8'))
            if saved.get('run_id') == config['run_id']:
                self.state.update(saved)
        self.state.update(interval=interval, polling=False)
        self.state['label'] = config.get('label', config['run_id'])
        self.state['branch_step'] = config.get('branch_step', 0)
        self.state['iteration_steps'] = config['iteration_steps']
        self.state['rollout_mode'] = config.get('rollout_mode', 'fixed_steps')
        self.state['branch_round'] = config.get('branch_round',
            self.state['branch_step'] / (config['iteration_steps'] or 1))
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

    def read_rank(self, rank):
        with self.lock:
            previous = next((s for s in self.state['ranks'] if s['rank'] == rank), {})
        c = self.config
        header = '\n'.join(f'{key}={value!r}' for key, value in dict(
            ROOT=c['root'], RUN=c['run_id'], RANK=rank, CURSOR=previous.get('cursor', {})).items())
        script = f"cd {shlex.quote(c['root'])}\n.venv/bin/python - <<'PY_MONITOR'\n{header}\n{REMOTE}\nPY_MONITOR\n"
        host = c['hosts'][rank] if 'hosts' in c else c['host']
        args = ['ssh', '-i', c['ssh_key'], '-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=yes',
                '-o', 'ConnectTimeout=8', '-o', 'ServerAliveInterval=10',
                *(['-o', 'UserKnownHostsFile=' + c['ssh_known_hosts']] if c.get('ssh_known_hosts') else []),
                '-p', str(c['ports'][rank]), host, 'bash -se']
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
            self.state.update(ranks=ranks, updated_at=time.time(), polling=False)
            payload = json.dumps(clean(self.state), ensure_ascii=False, allow_nan=False)
            self.output.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.output.with_suffix('.tmp')
            tmp.write_text(payload, encoding='utf-8')
            os.replace(tmp, self.output)

    def watch(self, max_hours):
        deadline = time.monotonic() + max_hours * 3600 if max_hours else math.inf
        while not self.stop.is_set() and time.monotonic() < deadline:
            self.poll()
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
    html = Path(__file__).with_name('dashboard_static') / 'training.html'

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if not allowed_client(self.client_address[0], networks):
                self.send_error(403)
                return
            if self.path == '/api/state':
                body, content = monitor.snapshot(), 'application/json; charset=utf-8'
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

        def log_message(self, *args):
            pass

    # Read-only HTTP: optional explicit LAN allowlist, no credentials/control API.
    server = ThreadingHTTPServer((str(listen), args.port), Handler)
    thread = threading.Thread(target=monitor.watch, args=(args.max_hours,), daemon=True)
    thread.start()
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
