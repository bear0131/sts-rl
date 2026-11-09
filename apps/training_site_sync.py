"""Publish read-only training charts from a remote host to its private Site.

The native monitor remains the single log parser. This process discovers a live
rank-zero continuous trainer, preserves prior runs and uploads bounded metrics.
No checkpoint, model, SSH key or inherited environment is sent to the Site.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time
import urllib.request
from urllib.parse import urlsplit

from apps.training_monitor import Monitor, clean


METRICS = frozenset('''global_timesteps rollout_round transitions_global
completed_full_games_global iteration_wall_sec rollout_sec rollout_sec_max
games_per_rollout_global
update_sec update_sec_max backward_sec sync_sec checkpoint_sec optimizer_steps
game/win_rate game/floor_mean game/episodes_total game/episodes_window
episode_rollout/completed_games_global episode_rollout/censored_games_global
episode_rollout/wins_global episode_rollout/floor_sum_global
upper/epochs upper/updates upper/critic_updates upper/total_sec_rank
upper/parameter_l2_change upper/policy_loss upper/value_loss upper/kl
upper/policy_pre/loss upper/policy_pre/ppo_loss upper/value_pre/mc_mse
upper/value_mc_mse_pre_update upper/policy_loss_pre_update
small_playout_train/loss small_playout_train/target_entropy
small_playout_train/target_kl small_playout_train/seconds
small_playout_train/teacher_top1 small_playout/turn_card_limit_decisions_global
'''.split())
PREFIXES = ('pure_rl/', 'upper/reward_', 'upper/campfire_', 'upper/epoch_',
            'upper/policy_pre/', 'upper/value_pre/', 'upper/kl_epoch_',
            'episode_rollout/')
RUN_FIELDS = ('root', 'run_id', 'label', 'iteration_steps', 'rollout_mode',
              'branch_step', 'branch_round', 'continuation_of', 'followed', 'status', 'note')


def compact_rows(rows):
    return [{key: clean(value) for key, value in row.items()
             if (key in METRICS or key.startswith(PREFIXES))
             and (value is None or isinstance(value, (bool, int, float)))}
            for row in rows]


def compact_run(run):
    return {**{key: run[key] for key in RUN_FIELDS if key in run},
            'rows': compact_rows(run.get('rows', []))}


def public_snapshot(state):
    result = compact_run(state)
    result.update({key: clean(state[key]) for key in (
        'interval', 'updated_at', 'polling', 'polling_ended', 'poll_error',
        'active_target', 'target_error', 'search_budget_rows') if key in state})
    result['comparisons'] = [compact_run(run) for run in state.get('comparisons', [])]
    result['ranks'] = [{key: clean(rank[key]) for key in (
        'rank', 'alive', 'step', 'progress', 'checked_at', 'attempted_at') if key in rank}
        | {'connection_error': '暂时无法连接这台机器' if rank.get('connection_error') else None}
        for rank in state.get('ranks', [])]
    # Resource telemetry is useful to GPT, but only the latest record is needed.
    result['resources'] = clean(state.get('resources', {}))
    latest = state.get('rows', [])[-1:]
    if latest and any(k in latest[0] for k in ('iteration_resources', 'rollout_resources')):
        result['resources'] = {key: clean(latest[0][key]) for key in
            ('iteration_resources', 'rollout_resources') if key in latest[0]}
    return result


def expanded_args(argv):
    result = []
    for value in argv:
        if value.startswith('@'):
            result.extend(Path(value[1:]).read_text().splitlines())
        else:
            result.append(value)
    return result


def argument(argv, name, default=None):
    values = [argv[i + 1] for i in range(len(argv) - 1) if argv[i] == name]
    return values[-1] if values else default


def discover_training(root, *, proc_root=Path('/proc')):
    """Select by live process start time, never by an old plan's timestamp."""
    candidates = []
    root = Path(root).resolve()
    for file in root.glob('*/training-plan*.json'):
        try:
            plan = json.loads(file.read_text())
            if plan.get('rank') != 0 or not plan.get('continuous'):
                continue
            run = Path(plan['run']).resolve()
            cwd = Path(plan['cwd']).resolve()
            if not run.is_relative_to(root) or not cwd.is_relative_to(root):
                continue
            pid = int((run / 'rank0.pid').read_text())
            proc = proc_root / str(pid)
            cmd = (proc / 'cmdline').read_bytes().split(b'\0')
            if b'rl.training.distributed_sb3' not in cmd or run.name.encode() not in cmd:
                continue
            if not (proc / 'cwd').resolve().is_relative_to(cwd):
                continue
            # After '(comm)' the first field is state (field 3), starttime is 22.
            fields = (proc / 'stat').read_text().rsplit(')', 1)[1].split()
            if fields[0] == 'Z':
                continue
            argv = expanded_args(plan['argv'])
            steps = int(argument(argv, '--n-envs')) * int(argument(argv, '--n-steps'))
            world = int(argument(argv, '--world-size'))
            target = dict(root=str(cwd), run_id=run.name, label=run.name,
                iteration_steps=steps * world, rollout_mode='fixed_steps',
                branch_step=0, branch_round=0, status='training',
                note='持续训练；曲线在完整训练轮次结束后更新。')
            candidates.append((int(fields[19]), target))
        except (OSError, ValueError, KeyError, IndexError, TypeError):
            continue
    return max(candidates, key=lambda item: item[0])[1] if candidates else None


def switch_monitor(config, path, interval, target):
    saved = json.loads(path.read_text()) if path.exists() else {}
    histories = list(saved.get('comparisons', []))
    if saved.get('run_id') and saved['run_id'] != target['run_id']:
        histories = [h for h in histories if h['run_id'] != saved['run_id']]
        histories.append(compact_run(saved))
        resumed = next((h for h in histories if h['run_id'] == target['run_id']), {})
        histories = [h for h in histories if h['run_id'] != target['run_id']]
        saved = {**target, 'rows': resumed.get('rows', []), 'ranks': []}
    monitor = Monitor({**config, **target, 'comparisons': []}, path, interval)
    monitor.state.update(saved)
    monitor.state.update(target)
    monitor.state.update(comparisons=histories, active_target=target,
                         polling=False, polling_ended=False, poll_error=None)
    return monitor


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, msg, headers, newurl):
        return None  # Never forward the Site bearer credential to another URL.


def upload(url, token, state, uploaded=None):
    parsed = urlsplit(url)
    if parsed.scheme != 'https' or not parsed.hostname or parsed.path not in ('', '/'):
        raise ValueError('publisher requires the exact HTTPS Site origin')
    def post(route, value):
        payload = json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(',', ':')).encode()
        if len(payload) > 24 * 1024 * 1024:
            raise ValueError('one monitor object exceeds the 24 MiB upload limit')
        request = urllib.request.Request(url.rstrip('/') + route, data=payload,
            headers={'Content-Type': 'application/json',
                     'OAI-Sites-Authorization': 'Bearer ' + token}, method='POST')
        with urllib.request.build_opener(NoRedirect).open(request, timeout=25) as response:
            result = json.load(response)
        if not result.get('ok'):
            raise ValueError('Site did not acknowledge the monitor snapshot')
        return result, payload

    uploaded = uploaded if uploaded is not None else {}
    snapshot = public_snapshot(state)
    index = []
    for run in snapshot['comparisons']:
        text = json.dumps(run, ensure_ascii=False, allow_nan=False, separators=(',', ':'))
        digest = hashlib.sha256(text.encode()).hexdigest()
        if not uploaded.get(digest):
            post('/api/ingest-history', dict(schema_version=1, sha256=digest, run_json=text))
            uploaded[digest] = True
        index.append({key: value for key, value in run.items() if key != 'rows'}
                     | dict(rows=[], row_count=len(run['rows']), history_key=digest))
    snapshot['comparisons'] = index
    result, payload = post('/api/ingest', dict(schema_version=1, state=snapshot))
    return dict(bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest(),
                received_at=result['received_at'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--state', type=Path, required=True)
    parser.add_argument('--credentials', type=Path, required=True)
    parser.add_argument('--poll-seconds', type=float, default=30)
    parser.add_argument('--max-polls', type=int, default=0)
    args = parser.parse_args()
    if not math.isfinite(args.poll_seconds) or args.poll_seconds < 10 or args.max_polls < 0:
        parser.error('poll-seconds must be >=10 and max-polls >=0 (0 means continuous)')
    config = json.loads(args.config.read_text())
    cache_file = args.state.with_suffix('.uploaded.json')
    uploaded = json.loads(cache_file.read_text()) if cache_file.exists() else {}
    target = {key: config[key] for key in RUN_FIELDS if key in config}
    monitor = switch_monitor(config, args.state, args.poll_seconds, target)
    attempts = 0
    while not args.max_polls or attempts < args.max_polls:
        started = time.monotonic()
        try:
            discovered = discover_training(config['discovery_root'])
            if discovered and discovered['run_id'] != monitor.config['run_id']:
                monitor = switch_monitor(config, args.state, args.poll_seconds, discovered)
            monitor.poll()
            monitor.state['poll_error'] = None
            monitor.state['resources'] = public_snapshot(monitor.state)['resources']
            # Persist only graph metrics between polls; retain private log cursors.
            monitor.state['rows'] = compact_rows(monitor.state['rows'])
            monitor.state['comparisons'] = [compact_run(h) for h in monitor.state['comparisons']]
            temporary = args.state.with_suffix('.tmp')
            temporary.write_text(json.dumps(clean(monitor.state), ensure_ascii=False, allow_nan=False))
            os.replace(temporary, args.state)
            credentials = json.loads(args.credentials.read_text())
            receipt = upload(config['site_url'], credentials['service_token'], monitor.state, uploaded)
            cache_file.write_text(json.dumps(uploaded))
            status = dict(ok=True, at=time.time(), run_id=monitor.config['run_id'],
                round=(monitor.state['rows'][-1].get('rollout_round') if monitor.state['rows'] else None), **receipt)
        except Exception as exc:
            # Do not log request headers, credentials or raw remote error bodies.
            status = dict(ok=False, at=time.time(), error_type=type(exc).__name__)
        args.state.with_suffix('.publisher.json').write_text(json.dumps(status))
        print(json.dumps(status), flush=True)
        attempts += 1
        if not args.max_polls or attempts < args.max_polls:
            time.sleep(max(1, args.poll_seconds - (time.monotonic() - started)))


if __name__ == '__main__':
    main()
